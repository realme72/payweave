from typing import Any

import pytest

from payweave.agent.mappers_gen import (
    assemble,
    check_submission,
    required_mappers,
    route_findings,
    sdk_reference,
    write_mapper,
)
from tests.agent.test_spec_card import FakeDocs, ScriptedLLM, tool

REFUND_BODY = """
def refund_body(amount: Money, reason: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"amount": amount.amount_minor}
    if reason:
        body["notes"] = {"reason": reason}
    return body
"""

REFUND = """
_REFUND_STATUS: dict[str, RefundStatus] = {
    "pending": RefundStatus.PENDING,
    "processed": RefundStatus.SUCCEEDED,
    "failed": RefundStatus.FAILED,
}


def refund(entity: Json, ctx: MapperContext) -> Refund:
    native = str(entity.get("status"))
    if native not in _REFUND_STATUS:
        raise MappingError(f"unmapped refund status {native!r}")
    status = _REFUND_STATUS[native]
    return Refund(
        gateway=ctx.gateway,
        gateway_refund_id=str(entity["id"]),
        gateway_attempt_id=str(entity["payment_id"]),
        status=status,
        amount=money(entity.get("amount"), entity.get("currency")),
        failure=FailureInfo(category=FailureCategory.UNKNOWN)
        if status is RefundStatus.FAILED
        else None,
        created_at=from_unix(entity.get("created_at")),
    )
"""


def test_sdk_reference_is_generated_from_the_code() -> None:
    ref = sdk_reference()
    assert "PaymentIntent(gateway: str, gateway_intent_id: str" in ref
    assert "attempts: tuple[PaymentAttempt, ...]" in ref
    assert "enum RefundStatus: PENDING='pending'" in ref


def test_good_code_passes_every_check() -> None:
    assert check_submission("refund_body", REFUND_BODY, {}) == []
    assert check_submission("refund", REFUND, {"refund_body": REFUND_BODY}) == []


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        ("import os\n" + REFUND_BODY, "don't write imports"),
        (
            REFUND_BODY.replace("(amount: Money", "(amt: Money").replace(
                "amount.amount_minor", "amt.amount_minor"
            ),
            "must take (amount, reason)",
        ),
        (REFUND_BODY + "\ndef helper() -> int:\n    return 1\n", "must be private"),
        ("LIMIT = 5\n" + REFUND_BODY, "constant 'LIMIT' must be private"),
        (REFUND_BODY.replace("return body", "return open('x')"), "sandbox"),
        (
            REFUND_BODY.replace('"amount": amount.amount_minor', '"amount": amount.nope'),
            "mypy line",
        ),
        ("def capture_body(attempt, amount):\n    return {}\n", "no function named refund_body"),
        ("def refund_body(:", "syntax error"),
    ],
)
def test_bad_submissions_are_explained(code: str, problem: str) -> None:
    problems = check_submission("refund_body", code, {})
    assert any(problem in p for p in problems), problems


def test_mypy_errors_point_at_the_submitted_code() -> None:
    bad = REFUND_BODY.replace('"amount": amount.amount_minor', '"amount": amount.nope')
    problems = check_submission("refund_body", bad, {"refund": REFUND})
    assert problems == ['mypy line 2 of your code: "Money" has no attribute "nope"  [attr-defined]']


def test_assembled_module_imports_the_sdk() -> None:
    module = assemble({"refund_body": REFUND_BODY})
    assert "from payweave.packages.sdk import (" in module
    assert "def refund_body" in module


def test_required_mappers_follow_the_manifest() -> None:
    manifest: dict[str, Any] = {"operations": {"get_payment": {}, "refund": {}}}
    assert required_mappers(manifest) == ["intent", "refund_body", "refund"]
    manifest["webhooks"] = {}
    assert "webhook_event" in required_mappers(manifest)


async def test_write_mapper_fixes_what_the_checks_report() -> None:
    llm = ScriptedLLM(
        tool("search_docs", {"query": "refund request body"}),
        tool("submit_code", {"code": "import os\n" + REFUND_BODY}),
        tool("submit_code", {"code": REFUND_BODY}),
    )
    result = await write_mapper(
        llm, FakeDocs(), "razorpay", "refund_body", manifest={}, facts={}, accepted={},
        typecheck=False,
    )  # fmt: skip
    assert result.code is not None
    assert result.code.startswith("def refund_body")
    assert result.searches == ["refund request body"]
    rejection = llm.requests[2].messages[-1].content
    assert "don't write imports" in rejection
    assert "Write the mapper" in llm.requests[0].messages[0].content
    assert "SDK reference" in llm.requests[0].messages[0].content


async def test_repair_prompt_carries_the_gate_findings() -> None:
    llm = ScriptedLLM(tool("submit_code", {"code": REFUND_BODY}))
    await write_mapper(
        llm, FakeDocs(), "razorpay", "refund_body", manifest={}, facts={}, accepted={},
        repair=("def refund_body(...): ...", ["refunds: step partial refund: requests[0].json"]),
        typecheck=False,
    )  # fmt: skip
    prompt = llm.requests[0].messages[0].content
    assert "This is a REPAIR" in prompt
    assert "requests[0].json" in prompt


def test_findings_are_routed_to_the_responsible_mapper() -> None:
    report = {
        "gates": {
            "G7": {
                "results": [
                    {
                        "scenario": "refunds",
                        "differences": [
                            "step partial refund: requests[0].json.speed: only in candidate",
                            "step partial refund: result.ok.status: reference 'succeeded' != "
                            "candidate 'pending'",
                        ],
                    },
                    {
                        "scenario": "manual_capture",
                        "differences": [
                            "step get authorized: result.ok.status: reference 'authorized' != "
                            "candidate 'requires_action'",
                            "step capture: requests[1].json.currency: only in reference",
                        ],
                        "crashes": [
                            "candidate step create manual: UnknownGatewayError: mapper "
                            "intent failed: KeyError: 'x'"
                        ],
                    },
                    {
                        "scenario": "webhooks",
                        "differences": ["step valid webhook: result.ok.event_type: x"],
                    },
                ]
            }
        }
    }
    routed = route_findings(report)
    assert sorted(routed) == ["capture_body", "intent", "refund", "refund_body", "webhook_event"]
    assert len(routed["intent"]) == 2


INTENT_STUB = """
def intent(primary: Json, related: Mapping[str, Json], ctx: MapperContext) -> PaymentIntent:
    raise MappingError("stub")
"""


async def test_build_package_writes_gates_and_repairs(tmp_path: Any) -> None:
    from payweave.agent.mappers_gen import build_package  # noqa: PLC0415

    (tmp_path / "manifest.toml").write_text(
        '[operations.get_payment.primary]\nmethod = "GET"\npath = "o/{intent_id}"\n'
        '[operations.refund]\nmethod = "POST"\npath = "p/{attempt_id}/refund"\n'
    )
    llm = ScriptedLLM(
        tool("submit_code", {"code": REFUND_BODY}),
        tool("submit_code", {"code": REFUND}),
        tool("submit_code", {"code": INTENT_STUB}),
        tool("submit_code", {"code": REFUND_BODY.replace("reason}", "reason, 'v': '2'}")}),
    )
    reports = [
        {
            "promotable": False,
            "failed_gates": ["G7"],
            "gates": {
                "G7": {
                    "results": [
                        {
                            "scenario": "refunds",
                            "differences": [
                                "step partial refund: requests[0].json.notes: only in reference"
                            ],
                        }
                    ]
                }
            },
        },
        {"promotable": True, "failed_gates": [], "gates": {}},
    ]

    async def gates(_draft: Any) -> dict[str, Any]:
        return reports.pop(0)

    out = await build_package(
        llm, FakeDocs(), "razorpay", tmp_path, facts={}, gates=gates, typecheck=False
    )
    assert out.promotable
    assert [r.name for r in out.results] == ["refund_body", "refund", "intent", "refund_body"]
    assert "This is a REPAIR" in llm.requests[3].messages[0].content
    module = (tmp_path / "mappers.py").read_text()
    assert "'v': '2'" in module  # the repaired version is what was assembled
    assert (tmp_path / "report.json").exists()
    assert sorted(p.name for p in (tmp_path / ".mappers").iterdir()) == [
        "intent.py",
        "refund.py",
        "refund_body.py",
    ]


async def test_build_package_resumes_from_saved_parts(tmp_path: Any) -> None:
    from payweave.agent.mappers_gen import build_package  # noqa: PLC0415

    (tmp_path / "manifest.toml").write_text(
        '[operations.refund]\nmethod = "POST"\npath = "p/{attempt_id}/refund"\n'
        '[operations.get_payment.primary]\nmethod = "GET"\npath = "o/{intent_id}"\n'
    )
    (tmp_path / ".mappers").mkdir()
    (tmp_path / ".mappers" / "refund_body.py").write_text(REFUND_BODY)
    (tmp_path / ".mappers" / "refund.py").write_text(REFUND)
    llm = ScriptedLLM(tool("submit_code", {"code": INTENT_STUB}))

    async def gates(_draft: Any) -> dict[str, Any]:
        return {"promotable": True, "failed_gates": [], "gates": {}}

    out = await build_package(
        llm, FakeDocs(), "razorpay", tmp_path, facts={}, gates=gates, typecheck=False
    )
    assert [r.name for r in out.results] == ["intent"]  # only the missing one
    assert out.promotable


async def test_plain_code_fallback_when_the_provider_rejects_the_forced_call() -> None:
    from payweave.llm import LLMBadRequestError  # noqa: PLC0415
    from tests.agent.test_spec_card import text  # noqa: PLC0415

    rejected = LLMBadRequestError("tool 'search_docs' was not in request.tools", provider="s")
    llm = ScriptedLLM(rejected, text("```python\n" + REFUND_BODY + "```"))
    result = await write_mapper(
        llm, FakeDocs(), "razorpay", "refund_body", manifest={}, facts={}, accepted={},
        typecheck=False, max_turns=1,
    )  # fmt: skip
    assert result.code is not None
    assert result.code.startswith("def refund_body")
    assert llm.requests[1].tools == ()  # the fallback offers no tools


async def test_saved_parts_are_rechecked_and_repaired(tmp_path: Any) -> None:
    from payweave.agent.mappers_gen import build_package  # noqa: PLC0415

    (tmp_path / "manifest.toml").write_text(
        '[operations.refund]\nmethod = "POST"\npath = "p/{attempt_id}/refund"\n'
        '[operations.get_payment.primary]\nmethod = "GET"\npath = "o/{intent_id}"\n'
    )
    parts = tmp_path / ".mappers"
    parts.mkdir()
    stale = REFUND_BODY.replace("if reason:", "if isinstance(reason, (str, float)):")
    (parts / "refund_body.py").write_text(stale)
    (parts / "refund.py").write_text(REFUND)
    (parts / "intent.py").write_text(INTENT_STUB)
    llm = ScriptedLLM(tool("submit_code", {"code": REFUND_BODY}))

    async def gates(_draft: Any) -> dict[str, Any]:
        return {"promotable": True, "failed_gates": [], "gates": {}}

    out = await build_package(
        llm, FakeDocs(), "razorpay", tmp_path, facts={}, gates=gates, typecheck=False
    )
    assert [r.name for r in out.results] == ["refund_body"]
    assert "builtin 'float'" in llm.requests[0].messages[0].content
    assert "float" not in (parts / "refund_body.py").read_text()


def test_problems_in_other_mappers_are_not_blamed_on_this_one() -> None:
    broken = REFUND_BODY.replace("if reason:", "if isinstance(reason, (str, float)):")
    assert check_submission("refund", REFUND, {"refund_body": broken}, typecheck=False) == []
    problems = check_submission("refund_body", broken, {"refund": REFUND}, typecheck=False)
    assert problems
    assert problems[0].startswith("sandbox: line 3: builtin 'float'")
