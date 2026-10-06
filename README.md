<h1 align="center">🧵 PayWeave</h1>

<p align="center">
  <b>Automated payment gateway integration: AI builds and tests each gateway, you get one API for all of them.</b><br>
  Hand it a gateway's docs and it builds a verified, secure integration that you then review and approve.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-✓-009688">
  <img alt="AI" src="https://img.shields.io/badge/AI-Claude%20%2B%20RAG%20%2B%20Agents-8A2BE2">
  <img alt="Status" src="https://img.shields.io/badge/status-planning-orange">
</p>

---

## What is PayWeave?

Integrating a new payment gateway usually means days of reading docs, mapping
fields, handling webhooks and writing tests. PayWeave automates that work:

1. **Read.** It takes in the gateway's API docs and OpenAPI spec using RAG.
2. **Plan.** An agent maps the gateway's operations onto PayWeave's common payment model.
3. **Generate.** It writes a declarative **Gateway Manifest**, and every field cites its source in the docs.
4. **Verify.** The manifest goes through schema checks, contract tests, real sandbox runs and your approval.
5. **Weave.** The approved gateway plugs into one unified API through the Strategy and Registry patterns.

> **Safety first:** the AI works only at build time and never runs in the payment path.
> Every generated integration has to pass deterministic checks before it can move money.

## Core ideas

- **One interface for every gateway:** `authorize`, `capture`, `charge`, `refund`, `void`, `get_status`, `verify_webhook`
- **Strategy + Registry:** gateways are swappable strategies that you look up by name
- **SOLID throughout:** small interfaces, dependency inversion, open for new gateways and closed for modification
- **Money done right:** integer minor units plus ISO-4217 currency codes, never floats
- **Secure by default:** secrets only from env, constant-time webhook signature checks, replay protection, idempotency

## Observability (OpenTelemetry)

Every gateway call is a span plus metrics (`payweave.gateway.operations`,
`payweave.gateway.operation.duration`), added by a decorator in
`payweave.observability`: gateways contain no telemetry code. Export is OTLP, so
the backend is configuration: Jaeger locally, Grafana Cloud or any OTLP backend later.

```bash
make jaeger        # local Jaeger v2, no Docker; UI at http://localhost:16686
make trace-demo    # a few TEST-mode calls, then look at service "payweave"
```

Only allowlisted fields are exported (gateway, operation, ids, amounts, statuses,
error class). Secrets, signatures, headers, metadata and error messages never are,
and `tests/observability` fails if any of them shows up in exported data.

## Status

Planning. See [PLAN.md](PLAN.md) for the architecture, decisions and phases.
