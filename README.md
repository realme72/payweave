<h1 align="center">🧵 PayWeave</h1>

<p align="center">
  <b>Automated payment gateway integration: AI builds and tests each gateway, you get one API for all of them.</b><br>
  Hand it a gateway's docs and it builds a verified, secure integration that you then review and approve.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="AI" src="https://img.shields.io/badge/AI-any%20OpenAI--compatible%20LLM%20%2B%20RAG-8A2BE2">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange">
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

## Quick start

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). Everything runs on free tiers.

```bash
git clone https://github.com/realme72/payweave && cd payweave
uv sync
cp .env.example .env    # then fill in what you use (test/sandbox keys only)
make check              # lint, types, tests, and verifies the committed packages
```

`make check` needs no keys. For the rest:

| To run | You need |
| --- | --- |
| The agent (`payweave.agent`) | A free LLM key: [Groq](https://console.groq.com/keys) or [Gemini](https://aistudio.google.com/apikey), or local [Ollama](https://ollama.com). Choose with `PAYWEAVE_LLM_PROVIDER`. |
| The knowledge base the agent searches | Postgres + [pgvector](https://github.com/pgvector/pgvector) (`PAYWEAVE_DATABASE_URL`), then `make docs-fetch` and `make kb-ingest GATEWAY=razorpay` |
| Sandbox gates and the trace demo | The gateway's **test-mode** keys in `.env` |

## Building a gateway

```bash
# 1. Extract the Spec Card (facts + citations) from the docs; scored if an answer key exists
uv run --env-file .env --extra knowledge python -m payweave.agent spec-card razorpay

# 2. Compile it into a draft package, then let the agent write and repair the mappers
uv run --env-file .env --extra knowledge python -m payweave.agent manifest razorpay .runs/spec-cards/<run>.json
uv run --env-file .env --extra knowledge python -m payweave.agent mappers <draft-dir>

# 3. Run the verification gates (writes report.json)
uv run --env-file .env python -m payweave.verify <package-dir>

# 4. Review and sign it yourself: the AI can never approve its own work
uv run python -m payweave.packages keygen <your-name>        # once
uv run python -m payweave.packages review <package-dir>
uv run python -m payweave.packages sign <package-dir> --signer <your-name>
```

`keygen` keeps your private key in `~/.config/payweave/keys` (`PAYWEAVE_KEY_DIR`) and adds
the public half to `packages/trusted_signers.toml`. The packages in this repo are signed
by the maintainer's key, which is why `verify` can check them anywhere.

## Status

Alpha. Working today:

- Hand-written Razorpay reference gateway, tested against the real test-mode sandbox.
- Gateway Package engine with gates G1 (strict types), G2 (sandbox), G5 (contract tests),
  G7 (differential against the reference) and G8 (human Ed25519 signature).
- An agent-written Razorpay package that matches the reference 10/10 on G7 and is signed.
- Cashfree as a holdout: the answer key was written before any agent run.

Next: citation gate (G3), OpenAPI checks (G4), more auth schemes, and the runtime
service. See [PLAN.md](PLAN.md) for the architecture, decisions and phases.

## Safety

Use **test / sandbox keys only**. Never put live keys in `.env`, and never commit `.env`
(it's gitignored; run `make secrets-scan` before you push). Gateway docs are fetched
locally and are not redistributed: only their URLs and hashes are committed.

## License

[MIT](LICENSE)
