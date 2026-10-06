"""Local Razorpay webhook receiver (TEST MODE ONLY), exposed through a zrok tunnel.

    set -a; . ./.env; set +a
    uv run --extra otel python scripts/webhook_receiver.py     # 127.0.0.1:8766
    zrok share public http://127.0.0.1:8766 --headless         # gives the public URL

POST /webhooks/razorpay  verifies the signature over the RAW body with
                         RazorpayGateway.verify_and_parse, dedupes on the event id,
                         answers 200 (valid) or 400 (invalid signature).
GET  /webhooks           JSON summary of everything received.

Each delivery is logged to .sandbox/webhooks.jsonl (gitignored) with the raw
body, so real payload samples can be compared with the docs (V2, V4).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from payweave.core.capabilities import WebhookHandler
from payweave.core.errors import PayWeaveError
from payweave.core.factory import GatewayFactory
from payweave.gateways.razorpay import RazorpayGateway
from payweave.observability import traced
from payweave.observability.setup import configure_telemetry

HOST, PORT = "127.0.0.1", 8766
LOG_FILE = Path(__file__).resolve().parent.parent / ".sandbox" / "webhooks.jsonl"
MAX_BODY = 1_000_000
SEEN: set[str] = set()
SUMMARY: list[dict[str, Any]] = []


def build_gateway() -> RazorpayGateway:
    factory = GatewayFactory.with_installed_providers()
    config = factory.config_from_env("razorpay")
    if not str(config.get("key_id", "")).startswith("rzp_test_"):
        sys.exit("Refusing to run: RAZORPAY_KEY_ID must be a TEST key (rzp_test_...)")
    if not config.get("webhook_secrets"):
        sys.exit("Set RAZORPAY_WEBHOOK_SECRETS in .env (the secret from the Razorpay dashboard)")
    gateway = factory.create("razorpay", config)
    if not isinstance(gateway, RazorpayGateway):
        sys.exit("expected a RazorpayGateway")
    return gateway


class Handler(BaseHTTPRequestHandler):
    gateway: WebhookHandler

    def _send(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/webhooks":
            self._send(200, SUMMARY)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/webhooks/razorpay":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            self._send(413, {"error": "too large"})
            return
        raw = self.rfile.read(length)  # the exact bytes Razorpay signed
        tracer = trace.get_tracer("payweave.webhooks")
        with tracer.start_as_current_span("POST /webhooks/razorpay", kind=SpanKind.SERVER) as span:
            status = self._handle(raw)
            span.set_attribute("http.response.status_code", status)
            if status != 200:
                span.set_status(StatusCode.ERROR, "rejected")

    def _handle(self, raw: bytes) -> int:
        headers = dict(self.headers.items())
        entry: dict[str, Any] = {
            "received_at": datetime.now(UTC).isoformat(),
            "razorpay_event": _peek_event(raw),
            "event_id": headers.get("X-Razorpay-Event-Id"),
        }
        try:
            event = self.gateway.verify_and_parse(raw, headers)
        except PayWeaveError as exc:
            entry |= {"signature": "INVALID", "error": type(exc).__name__}
            status = 400
        else:
            duplicate = event.event_id in SEEN
            SEEN.add(event.event_id)
            entry |= {
                "signature": "valid",
                "duplicate": duplicate,
                "payweave_event": str(event.event_type),
                "gateway_intent_id": event.gateway_intent_id,
                "gateway_refund_id": event.gateway_refund_id,
            }
            status = 200
        SUMMARY.append(entry)
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(json.dumps({**entry, "raw_body": raw.decode("utf-8", "replace")}) + "\n")
        print(json.dumps(entry), flush=True)
        trace.get_current_span().set_attributes(
            {
                f"webhook.{k}": v
                for k, v in entry.items()
                if k != "received_at" and isinstance(v, str | bool)
            }
        )
        self._send(status, {"ok": status == 200})
        return status

    def log_message(self, format: str, *args: object) -> None:
        pass


def _peek_event(raw: bytes) -> str | None:
    """For the log only, so even rejected deliveries show what they claimed to be."""
    try:
        event = json.loads(raw).get("event")
    except (ValueError, AttributeError):
        return None
    return event if isinstance(event, str) else None


def main() -> None:
    # Traces go to Jaeger (make jaeger) as service "payweave-webhooks".
    shutdown = configure_telemetry(service_name="payweave-webhooks", instrument_http=False)
    gateway = traced(build_gateway())
    assert isinstance(gateway, WebhookHandler)  # noqa: S101 - startup sanity check
    Handler.gateway = gateway
    print(f"Webhook receiver on http://{HOST}:{PORT}/webhooks/razorpay", flush=True)
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
