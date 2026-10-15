"""Try a Cashfree package against the Cashfree sandbox (TEST MODE ONLY, localhost only).

    uv run --env-file .env python scripts/cashfree_checkout.py <package-dir>
    # then open http://127.0.0.1:8766

Needs CASHFREE_KEY_ID and CASHFREE_KEY_SECRET (sandbox keys) in .env. The package's
live guard refuses anything but a test secret, and test keys go to sandbox.cashfree.com.

Every call goes through PayWeave (the package's manifest + mappers), not a Cashfree SDK:
create an order, pay it in Cashfree's sandbox checkout, then fetch it, refund part of
it and fetch the refund. Each step's result is shown and saved to
.sandbox/cashfree_results.json. The secret stays in this process; the browser only
gets the order's payment_session_id.

Test card (from Cashfree's test data): 4706 1312 1121 2123, expiry 03/28, CVV 123,
name Test, OTP 111000. Or UPI: testsuccess@gocash.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from payweave.core.errors import GatewayError
from payweave.core.factory import GatewayFactory
from payweave.core.models import CaptureMode, CreatePaymentRequest, Customer, RefundRequest
from payweave.core.money import Money
from payweave.packages.runtime import PackageProvider, load_package

HOST, PORT = "127.0.0.1", 8766
RESULTS_FILE = Path(__file__).resolve().parent.parent / ".sandbox" / "cashfree_results.json"
LOOP = asyncio.new_event_loop()


def gateway(package_dir: Path) -> Any:
    package = load_package(package_dir)
    config = {
        "key_id": os.environ.get("CASHFREE_KEY_ID", ""),
        "key_secret": os.environ.get("CASHFREE_KEY_SECRET", ""),
    }
    # allow_unsigned: a draft the agent just wrote, tried in the sandbox before anyone signs it.
    return GatewayFactory([PackageProvider(package, allow_unsigned=True)]).create(
        "cashfree", config
    )


def record(step: str, outcome: Any) -> dict[str, Any]:
    entry = {"step": step, "outcome": outcome}
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps([*results, entry], indent=2, default=str))
    print(json.dumps(entry, default=str), flush=True)
    return entry


def run(step: str, coro: Any) -> tuple[Any, dict[str, Any]]:
    try:
        value = LOOP.run_until_complete(coro)
    except GatewayError as exc:
        return None, record(step, {"error": type(exc).__name__, "message": str(exc)})
    return value, record(step, value.model_dump(mode="json"))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PayWeave x Cashfree</title>
<script src="https://sdk.cashfree.com/js/v3/cashfree.js"></script></head>
<body style="font-family:sans-serif;max-width:820px;margin:32px auto">
<h2>PayWeave x Cashfree sandbox (TEST MODE)</h2>
<p>Card <code>4706 1312 1121 2123</code>, expiry <code>03/28</code>, CVV <code>123</code>,
name <code>Test</code>, OTP <code>111000</code> &middot; or UPI <code>testsuccess@gocash</code>.</p>
{body}
<h3>Last result</h3><pre style="background:#f4f4f4;padding:12px;white-space:pre-wrap">{result}</pre>
</body></html>"""


def page(body: str, result: Any = "") -> bytes:
    text = json.dumps(result, indent=2, default=str) if result != "" else "(none yet)"
    return PAGE.format(body=body, result=html.escape(text)).encode()


def actions(intent_id: str) -> str:
    q = f"intent={html.escape(intent_id)}"
    return (
        f"<p>Order <code>{html.escape(intent_id)}</code>: "
        f'<a href="/status?{q}">1. check status</a> &middot; '
        f'<a href="/refund?{q}&amount=5000">2. refund &#8377;50</a> &middot; '
        f'<a href="/refund?{q}">3. refund the rest</a> &middot; '
        f'<a href="/">new order</a></p>'
    )


class Handler(BaseHTTPRequestHandler):
    gw: Any = None
    last_refund: dict[str, str] = {}  # noqa: RUF012 - one local tester

    def send(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            form = (
                '<form action="/new">Customer phone (Cashfree requires one; type a test '
                'number yourself): <input name="phone" pattern="[0-9]{10}" required> '
                "<button>Create a &#8377;100 test order</button></form>"
            )
            self.send(page(form))
        elif url.path == "/new":
            self.new_order(q.get("phone", ""))
        elif url.path == "/status":
            _, entry = run("get_payment", self.gw.get_payment(q["intent"]))
            self.send(page(actions(q["intent"]), entry["outcome"]))
        elif url.path == "/refund":
            amount = Money.of(int(q["amount"]), "INR") if "amount" in q else None
            request = RefundRequest(
                gateway_intent_id=q["intent"],
                amount=amount,
                idempotency_key=f"pw-{uuid.uuid4().hex[:20]}",
                reason="PayWeave sandbox test",
            )
            refund, entry = run("refund", self.gw.refund(request))
            body = actions(q["intent"])
            if refund is not None:
                rid = refund.gateway_refund_id
                link = f"/get_refund?intent={q['intent']}&refund={rid}"
                body += f'<p><a href="{link}">4. fetch refund {html.escape(rid)}</a></p>'
            self.send(page(body, entry["outcome"]))
        elif url.path == "/get_refund":
            _, entry = run(
                "get_refund",
                self.gw.get_refund(q["refund"], gateway_intent_id=q["intent"]),
            )
            self.send(page(actions(q["intent"]), entry["outcome"]))
        else:
            self.send(b"not found", 404)

    def new_order(self, phone: str) -> None:
        try:
            customer = Customer(id="payweave-tester", phone=phone)
        except ValueError:
            self.send(page('<p>Enter a 10-digit phone. <a href="/">back</a></p>'))
            return
        request = CreatePaymentRequest(
            amount=Money.of(10000, "INR"),
            idempotency_key=f"pw{uuid.uuid4().hex[:20]}",
            merchant_reference="payweave-sandbox",
            capture_mode=CaptureMode.AUTOMATIC,
            customer=customer,
        )
        intent, entry = run("create_payment", self.gw.create_payment(request))
        if intent is None:
            self.send(page('<p><a href="/new">try again</a></p>', entry["outcome"]))
            return
        params = intent.next_action.params if intent.next_action else {}
        session = params.get("payment_session_id")
        if not session:
            note = "<p>The package gave no payment_session_id to open checkout with.</p>"
            self.send(page(note + actions(intent.gateway_intent_id), entry["outcome"]))
            return
        script = (
            '<button id="pay">Pay with Cashfree (sandbox)</button><script>'
            'const cf = Cashfree({mode: "sandbox"});'
            'document.getElementById("pay").onclick = () => cf.checkout('
            f'{{paymentSessionId: {json.dumps(session)}, redirectTarget: "_modal"}})'
            '.then(r => { document.getElementById("after").style.display = "block"; });'
            "</script>"
            '<div id="after" style="display:none"><p>Checkout closed.</p></div>'
        )
        self.send(page(script + actions(intent.gateway_intent_id), entry["outcome"]))

    def log_message(self, *_: Any) -> None:
        pass


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    Handler.gw = gateway(Path(sys.argv[1]))
    print(f"open http://{HOST}:{PORT}  (results -> {RESULTS_FILE})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
