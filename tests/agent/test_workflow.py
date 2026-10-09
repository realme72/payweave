from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from payweave.agent.workflow import Deps, build_graph
from tests.agent.test_manifest_compiler import c, none, razorpay_card
from tests.agent.test_spec_card import FakeDocs, ScriptedLLM, fact, tool

CONFIG: Any = {"configurable": {"thread_id": "t1"}}


async def start(tmp_path: Path, card: dict[str, Any], llm: Any = None) -> Any:
    app = build_graph(
        Deps(drafts=tmp_path / "draft", llm=llm, docs=FakeDocs() if llm else None), InMemorySaver()
    )
    await app.ainvoke({"gateway": "razorpay", "card": card}, CONFIG)
    return app


async def test_pauses_for_a_person_then_writes_the_draft(tmp_path: Path) -> None:
    app = await start(tmp_path, razorpay_card())
    paused = await app.aget_state(CONFIG)
    assert paused.next == ("confirm_card",)
    shown = paused.tasks[0].interrupts[0].value
    assert "endpoints.capture = 'POST /v1/payments/:id/capture'" in "\n".join(shown["card"])
    assert not (tmp_path / "draft").exists()  # nothing written before approval

    result = await app.ainvoke(Command(resume={"approved": True, "by": "alice"}), CONFIG)
    assert result["status"] == "drafted"
    assert (tmp_path / "draft" / "manifest.toml").read_text().startswith("schema_version = 1")
    assert '"by": "alice"' in (tmp_path / "draft" / "provenance.json").read_text()


async def test_rejected_card_writes_nothing(tmp_path: Path) -> None:
    app = await start(tmp_path, razorpay_card())
    result = await app.ainvoke(Command(resume={"approved": False, "by": "alice"}), CONFIG)
    assert result["status"] == "card_rejected"
    assert not (tmp_path / "draft").exists()


async def test_gaps_mark_the_run_for_a_person(tmp_path: Path) -> None:
    app = await start(tmp_path, razorpay_card(ids={"refund_id_example": none()}))
    result = await app.ainvoke(Command(resume={"approved": True, "by": "alice"}), CONFIG)
    assert result["status"] == "needs_person"
    assert any("ids.refund" in g for g in result["compiled"]["gaps"])


async def test_only_missing_sections_are_extracted(tmp_path: Path) -> None:
    card = razorpay_card(errors=None)
    llm = ScriptedLLM(
        tool("submit_errors", {"error_code_field": fact(None), "reason_field": fact(None)})
    )
    app = await start(tmp_path, card, llm)
    assert len(llm.requests) == 1  # only `errors` was asked for
    state = (await app.aget_state(CONFIG)).values
    assert state["card"]["errors"]["error_code_field"]["value"] is None
    assert state["extraction"][0]["section"] == "errors"


async def test_full_run_writes_mappers_and_gates(tmp_path: Path) -> None:
    from tests.agent.test_mappers_gen import INTENT_STUB, REFUND, REFUND_BODY  # noqa: PLC0415

    card = razorpay_card(capture={"manual_capture": c(False)})

    def stub(name: str, params: str) -> Any:
        return tool("submit_code", {"code": f"def {name}({params}) -> Any:\n    return {{}}\n"})

    llm = ScriptedLLM(
        tool("submit_code", {"code": REFUND_BODY}),
        stub("create_payment_body", "request: CreatePaymentRequest, settings: Mapping[str, Any]"),
        tool("submit_code", {"code": REFUND}),
        stub("webhook_event", "body: Json, event_id: str, ctx: MapperContext"),
        tool("submit_code", {"code": INTENT_STUB}),
    )

    async def gates(_draft: Any) -> dict[str, Any]:
        return {"promotable": True, "failed_gates": [], "gates": {}}

    deps = Deps(drafts=tmp_path / "draft", llm=llm, docs=FakeDocs(), gates=gates)
    app = build_graph(deps, InMemorySaver())
    await app.ainvoke({"gateway": "razorpay", "card": card}, CONFIG)
    result = await app.ainvoke(Command(resume={"approved": True, "by": "alice"}), CONFIG)
    assert result["status"] == "promotable", result.get("compiled", {}).get("gaps")
    assert [m["name"] for m in result["mappers"]["mappers"]] == [
        "refund_body",
        "create_payment_body",
        "refund",
        "webhook_event",
        "intent",
    ]
    assert (tmp_path / "draft" / "mappers.py").exists()
