"""Local Razorpay Checkout test page (TEST MODE ONLY, localhost only).

    set -a; . ./.env; set +a
    uv run python scripts/checkout_harness.py            # http://127.0.0.1:8765

GET  /                 scenario list (scripts/razorpay_scenarios.py), one link each
GET  /new?scenario=<id>  creates an order for that scenario + Pay button
POST /return            verifies the checkout signature server-side, records the outcome
GET  /results           JSON log, also saved to .sandbox/scenario_results.json

Then: uv run python scripts/verify_scenarios.py

Only the public key_id reaches the browser. The secret stays in this process.
"""

from __future__ import annotations

import asyncio
import html
import json
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from razorpay_scenarios import SCENARIOS

from payweave.core.errors import InvalidSignatureError
from payweave.core.factory import GatewayFactory
from payweave.core.models import CaptureMode, CreatePaymentRequest
from payweave.core.money import Money
from payweave.gateways.razorpay import RazorpayGateway

HOST, PORT = "127.0.0.1", 8765
RESULTS_FILE = Path(__file__).resolve().parent.parent / ".sandbox" / "scenario_results.json"
RESULTS: list[dict[str, Any]] = (
    json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
)


def _record(entry: dict[str, Any]) -> None:
    RESULTS.append(entry)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(RESULTS, indent=2))
    print(json.dumps(entry), flush=True)


INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>PayWeave scenarios</title></head>
<body style="font-family:sans-serif;max-width:760px;margin:40px auto">
<h2>PayWeave x Razorpay test scenarios (TEST MODE)</h2>
<p><b>Contact step:</b> choose <b>skip OTP</b>. Never enter a real phone number.
Cards: any CVV, any future expiry. Each link creates a fresh order.</p>
<table cellpadding="6" style="border-collapse:collapse">{rows}</table></body></html>"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PayWeave test checkout</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script></head>
<body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>PayWeave test checkout (TEST MODE)</h2>
<p><b>{title}</b><br>{steps}</p>
<p>Order <code id="order">{order_id}</code> &middot; {amount} &middot; capture: {mode}</p>
<button id="pay" style="font-size:18px;padding:10px 24px">Pay</button>
<pre id="out"></pre>
<script>
const out = document.getElementById("out");
function report(kind, payload) {{
  fetch("/return", {{method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{kind, payload, order_id: "{order_id}"}})}}).then(r => r.json())
    .then(j => {{ out.textContent += kind + ": " + JSON.stringify(j) + "\\n"; }});
}}
const rzp = new Razorpay({{
  key: "{key_id}", order_id: "{order_id}", name: "PayWeave test", description: "sandbox test",
  // No phone number at all: Checkout sends REAL SMS OTPs to whatever number is entered.
  hidden: {{contact: true}}, remember_customer: false,
  prefill: {{email: "test@example.com"}},
  handler: (resp) => report("success", resp),
  modal: {{ondismiss: () => report("dismissed", {{}})}},
}});
rzp.on("payment.failed", (resp) => report("failed", resp.error));
document.getElementById("pay").onclick = () => rzp.open();
</script></body></html>"""


def build_gateway(**overrides: object) -> RazorpayGateway:
    factory = GatewayFactory.with_installed_providers()
    config: dict[str, object] = {**factory.config_from_env("razorpay"), **overrides}
    if not str(config.get("key_id", "")).startswith("rzp_test_"):
        sys.exit("Refusing to run: RAZORPAY_KEY_ID must be a TEST key (rzp_test_...)")
    gateway = factory.create("razorpay", config)
    if not isinstance(gateway, RazorpayGateway):
        sys.exit("expected a RazorpayGateway")
    return gateway


async def _create(request: CreatePaymentRequest, expiry: int | None) -> Any:
    # A fresh gateway per event loop: an httpx client can't be shared across loops.
    overrides = {"manual_capture_expiry_minutes": expiry} if expiry else {}
    gateway = build_gateway(**overrides)
    try:
        return await gateway.create_payment(request)
    finally:
        await gateway.aclose()


class Handler(BaseHTTPRequestHandler):
    gateway: RazorpayGateway

    def _send(self, status: int, body: str, ctype: str = "text/html") -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            rows = "".join(
                f"<tr><td><a href='/new?scenario={sc.id}'>{sc.id}</a></td>"
                f"<td><b>{html.escape(sc.title)}</b><br>{html.escape(sc.steps)}</td>"
                f"<td>{html.escape(sc.note)}</td></tr>"
                for sc in SCENARIOS.values()
            )
            self._send(200, INDEX.format(rows=rows))
        elif url.path == "/new":
            sc = SCENARIOS.get(query.get("scenario", [""])[0])
            if sc is None:
                self._send(404, "unknown scenario")
                return
            amount = Money.of(int(query.get("amount", ["20000"])[0]), "INR")
            request = CreatePaymentRequest(
                amount=amount,
                merchant_reference=f"pw-{sc.id}-{secrets.token_hex(4)}",
                idempotency_key=f"pw_harness_{secrets.token_hex(12)}",
                capture_mode=CaptureMode(sc.mode),
            )
            intent = asyncio.run(_create(request, sc.expiry))
            params = intent.next_action.params  # type: ignore[union-attr]
            _record({"kind": "created", "scenario": sc.id, "order_id": params["order_id"]})
            self._send(
                200,
                PAGE.format(
                    key_id=html.escape(params["key_id"]),
                    order_id=html.escape(params["order_id"]),
                    amount=html.escape(str(amount)),
                    mode=sc.mode,
                    title=html.escape(f"{sc.id}: {sc.title}"),
                    steps=html.escape(sc.steps),
                ),
            )
        elif url.path == "/results":
            self._send(200, json.dumps(RESULTS, indent=2), "application/json")
        else:
            self._send(404, "not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/return":
            self._send(404, "not found")
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        kind, payload = body.get("kind"), body.get("payload") or {}
        entry: dict[str, Any] = {"kind": kind, "order_id": body.get("order_id"), "payload": payload}
        if kind == "success":
            try:
                entry["verified_order_id"] = self.gateway.verify_checkout_return(payload)
                entry["signature"] = "valid"
            except InvalidSignatureError:
                entry["signature"] = "INVALID"
        _record(entry)
        self._send(
            200, json.dumps({k: v for k, v in entry.items() if k != "payload"}), "application/json"
        )

    def log_message(self, format: str, *args: object) -> None:  # quiet default access log
        pass


def main() -> None:
    Handler.gateway = build_gateway()
    print(f"PayWeave test scenarios on http://{HOST}:{PORT}/", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
