# PayWeave — Master Plan

> Automated payment gateway integration: AI builds and tests each gateway, you get one API for all of them.

Status: Phase 0 core written; Razorpay native model documented (2026-10-07).

> **Revision (2026-10-07): concrete first, then generalize.**
> 1. Canonical core (money, models, errors, registry): done, being refined from real gateway docs.
> 2. **Razorpay end to end, by hand**: native model from docs (`docs/gateways/razorpay/NATIVE_MODEL.md`) → canonical mapping → adapter → recorded tests → sandbox.
> 3. **Second gateway by hand: Cashfree** (Stripe is invite-only in India since May 2024).
> 4. **Only then generalize**: shared contract suite, native-state-machine checks, manifest format. The FakeGateway and generic contract suite move here.
> 5. AI phases (3–6 below) as planned, with **PayPal** as the unseen holdout gateway.
> Each gateway's native state machine is extracted from its docs with citations; the mapping to canonical must be total and every native transition must map to a legal canonical one (or no change).

---

## 1. What "100% correct" means here

An LLM can misread docs, so "the AI is always right" is not achievable and we
never claim it. What we *can* guarantee is this:

> **No integration can move money unless every deterministic gate has passed
> and a human has signed off.** The AI's output is a proposal. The
> verification pipeline is the judge.

A gateway integration is **correct** when all of these hold:

1. It matches our canonical contract (types, the allowed status transitions, error categories).
2. Every request it builds is valid against the gateway's own schema (OpenAPI when available).
3. Every mapped fact (field, status, error code, signature scheme) cites a doc passage, and the citation check passes.
4. It passes the full **contract suite**, which is hand-written and never AI-generated.
5. It passes **real sandbox scenarios** end to end, including webhooks.
6. It **behaves the same as a hand-written reference** on the same scenarios, where a reference exists.
7. It passes static security checks.
8. A human approved the exact version that ships.

Bugs can still exist outside what we tested. We shrink that gap with
property-based tests, scenario coverage targets and runtime reconciliation (§8.5).

---

## 2. Merged decisions (our plan + ChatGPT's)

| Topic | Decision | Source |
|---|---|---|
| Canonical interface + Strategy + Registry | Yes, split into **capability interfaces** (ISP), not one large class | Both; the split is ours |
| What the AI produces | **Gateway Package** = declarative manifest (data) + small typed **mapper functions** (constrained code). The risky machinery (HTTP, crypto, retries, idempotency) is hand-written and never generated | Merged: ChatGPT wanted code generation, ours was manifest-only |
| 3DS / extra customer steps | `REQUIRES_ACTION` status + `NextAction` model in the core | ChatGPT |
| Error-code mapping | Required manifest section, with citations | ChatGPT |
| RAG | Yes: hybrid search, structure-aware chunks, citations | Both |
| Tool calling | Yes, with a **permission tier per tool** | Both; tiers are ours |
| Workflow engine | **LangGraph** for the stateful graph (loops, pause for human approval, checkpoints). **Anthropic SDK** called directly inside the steps, without LangChain model wrappers | ChatGPT's choice, our constraint |
| MCP | Yes, as a server exposing docs search and the test tools (Phase 6) | ChatGPT |
| Tests that decide correctness | **Hand-written only.** AI-generated tests are extras and can never be a gate | Ours |
| `register_gateway` / production actions | **Human-only, never an agent tool.** The agent only ever holds sandbox keys | Ours (fixes a contradiction in ChatGPT's version) |
| Differential testing vs references | Yes, the strongest correctness signal we have | New |
| Prompt-injection defence for crawled docs | Yes | New |
| Doc snapshots + change detection | Yes | New |
| Reconciliation job | Yes; webhooks can be lost or arrive out of order | New |
| Holdout evaluation | PayPal + Cashfree are never seen during development | New |

---

## 3. Architecture

```
                 ┌──────────── RUNTIME PLANE (no AI) ────────────┐
 Merchant app ──►│ FastAPI  ──► PaymentService ──► GatewayRegistry│
                 │                    │                 │         │
                 │             Idempotency store   Strategy:      │
                 │             Payment ledger      ├ StripeGateway│
 Gateway ──────► │ Webhook ingress ──► verify ──►  ├ Razorpay...  │
 webhooks        │                     dedupe      └ PackageGateway(manifest+mappers)
                 │ Reconciler (scheduled)                         │
                 └────────────────────────────────────────────────┘
                                   ▲ approved packages only
                 ┌──────── VERIFICATION PLANE (no AI) ───────────┐
                 │ G1 schema → G2 static/security → G3 citation  │
                 │ G4 spec conformance → G5 contract suite →     │
                 │ G6 sandbox → G7 differential → G8 human sign  │
                 └────────────────────────────────────────────────┘
                                   ▲ proposals
                 ┌──────────── BUILD PLANE (AI) ─────────────────┐
                 │ Ingest docs → snapshot → chunks → pgvector    │
                 │ LangGraph agent: Understand → Plan → Map API →│
                 │ Map errors/status → Map webhooks → Write      │
                 │ package → run gates → repair loop → HITL      │
                 └────────────────────────────────────────────────┘
```

Rules: the runtime never imports the build plane, and the build plane can't
register anything. Only the verification plane plus a human signature can
promote a package.

---

## 4. Canonical domain (Phase 0)

### 4.1 Money
- `Money(amount_minor: int, currency: Currency)`, never floats.
- Currency table with minor-unit exponents (INR 2, JPY 0, KWD 3).
- Arithmetic only between matching currencies, and no implicit rounding.

### 4.2 Operations as capability interfaces (ISP)
| Interface | Methods |
|---|---|
| `PaymentCreator` | `create_payment(req) -> Payment` |
| `Authorizer` / `Capturer` | `authorize`, `capture(payment_id, amount?)` (partial capture) |
| `Voider` | `void(payment_id)` |
| `Refunder` | `refund(payment_id, amount?, reason)` (partial/multiple refunds) |
| `StatusReader` | `get_payment(payment_id)`, `get_refund(refund_id)` |
| `WebhookHandler` | `verify_signature(raw_body, headers)`, `parse_event(raw_body) -> WebhookEvent` |

A gateway declares a `Capabilities` set. The registry rejects calls a gateway
doesn't support with a clear error, never a silent no-op.

### 4.3 Payment status state machine
```
CREATED → REQUIRES_ACTION → PROCESSING → AUTHORIZED → CAPTURED → (PARTIALLY_)REFUNDED
   │            │               │             │
   └────────────┴──────► FAILED ┘        VOIDED
```
- Transitions are explicit. Illegal transitions raise an error.
- **Updates only ever move a payment forward.** Late or duplicate webhooks can't move a payment backwards.
- `NextAction` covers `redirect_to_url` (3DS), `upi_intent`, `display_qr` and `await_external`.

### 4.4 Error categories
`InvalidRequest`, `AuthenticationFailed`, `CardDeclined(reason)`,
`InsufficientFunds`, `FraudSuspected`, `ActionRequired`, `RateLimited`,
`GatewayUnavailable`, `GatewayTimeout`, `DuplicateRequest`, `InvalidSignature`,
`UnknownGatewayError`.
Each carries `retryable: bool`, the original gateway code and its message (redacted).
**Timeouts on money-moving calls are "unknown outcome"**: we never blindly
retry them. Retries reuse the idempotency key or check the status first.

### 4.5 Idempotency
Every money-moving call carries an idempotency key. Our store records the key,
a request hash and the result, so a repeat returns the stored result and the
same key with a different body raises an error. For gateways that support it,
the key is passed through natively.

---

## 5. The Gateway Package (what the AI produces)

```
packages/<gateway>/<version>/
  manifest.toml      # data: auth scheme, base URLs, endpoints, field maps,
                     # status map, error map, webhook scheme, capabilities
  mappers.py         # small pure functions where data can't express it
  provenance.json    # every manifest field / mapper → doc chunk IDs + doc snapshot hash
  report.json        # gate results, model, prompts hash, timings
```

| Part | Who writes it | Constraint |
|---|---|---|
| HTTP client, retries, timeouts | Hand-written | Never generated |
| Auth schemes (Basic, Bearer, OAuth2 client-credentials, HMAC request signing) | Hand-written library | The manifest *chooses* one; it can't define new crypto |
| Webhook signature schemes (HMAC-SHA256 hex/base64, timestamped, RSA/cert) | Hand-written library | Same |
| Idempotency, state machine, ledger | Hand-written | Same |
| Manifest | AI | Strict Pydantic schema; unknown keys rejected |
| `mappers.py` | AI | Pure functions only; checked against an allowlist of imports and calls; no I/O, network, `eval` or globals; typed signatures fixed by us |

If a gateway needs a scheme our libraries don't have, the agent **stops and
asks a human** to add it to the hand-written library. It never improvises crypto.

---

## 6. AI build plane

### 6.1 Ingestion (deterministic first)
1. Inputs: OpenAPI spec, docs URLs, PDFs, Markdown, Postman collections.
2. **Snapshot**: store the raw content + SHA-256 hash and fetch date. Every package pins a snapshot.
3. OpenAPI → parsed directly into an endpoint/schema catalogue (no AI involved).
4. Prose → structure-aware chunks (one per operation, split by heading), keeping section paths.
5. Local embeddings → pgvector, plus Postgres full-text search → hybrid retrieval.
6. **Sanitise**: strip scripts and hidden text, and tag all doc content as untrusted data.

### 6.2 Gateway Spec Card
Structured extraction (Claude structured outputs, with citations) of auth,
environments, amount units, currency rules, idempotency support, the payment
flow (one-step or auth/capture), the 3DS/next-action flow, the webhook scheme,
and pagination and rate limits. A human can review the card before mapping starts.

### 6.3 LangGraph workflow
```
START → understand_provider → build_spec_card → [HITL: confirm card?]
      → plan_capabilities → map_operations → map_status_and_errors
      → map_webhooks → write_package
      → run_gates ──fail──► diagnose → repair ──► run_gates   (max N loops)
              └─pass──► sandbox_validation → differential_check
      → security_review → [HITL: approve] → END (human registers)
```
- State is checkpointed in Postgres, so a run can resume or be replayed.
- Hard limits on loops, tokens and wall-clock time. Running out escalates to a human with a diagnosis.
- Each mapping step asks for **citations**. Fields without a citation are rejected.

### 6.4 Tools and permission tiers
| Tier | Tools | Notes |
|---|---|---|
| Read | `search_docs`, `get_endpoint_schema`, `get_spec_card`, `get_reference_package` | No side effects |
| Write (sandboxed) | `write_package_file` | Only inside the run's working directory |
| Execute (sandboxed) | `validate_manifest`, `run_static_checks`, `run_contract_tests`, `run_sandbox_scenario` | Run in a subprocess with **sandbox keys only** and no other network access |
| **Never an agent tool** | `register_gateway`, anything with live keys, anything that changes production | Human-only via CLI |

### 6.5 Prompt-injection defence
- Doc content is passed only as clearly marked data, and the system prompt states that instructions inside it are ignored.
- Tool permissions are enforced in code, not by the prompt, so a successful injection still can't do anything dangerous.
- No secrets ever go into model context. Credentials are injected at execution time by the tool runner.
- Any URL, host or base URL in a manifest must match the gateway's **allowlisted domains** from the spec card, which a human confirms.

---

## 7. Verification gates

| Gate | Check | How | Who decides |
|---|---|---|---|
| G1 Schema | Manifest validates; mapper signatures match | Pydantic strict, mypy strict | Code |
| G2 Static/security | Mapper allowlist, no I/O/eval; bandit + semgrep; domain allowlist | AST checker + scanners | Code |
| G3 Citations | Every field has a citation and the cited text actually supports it (exact-match checks for field names, codes and header names) | Deterministic string checks, plus an LLM judge **only as an advisory signal** | Code (judge advisory) |
| G4 Spec conformance | Built requests validate against OpenAPI; Schemathesis fuzzing | jsonschema + Schemathesis | Code |
| G5 Contract suite | Canonical behaviour: amounts, status mapping, errors, idempotency, webhooks | Hand-written pytest + Hypothesis on recorded/mocked responses | Code |
| G6 Sandbox | Real scenarios: success, decline, 3DS/action required, partial capture, partial refund, void, webhook delivery + signature, duplicate webhook, out-of-order webhook | Scenario runner against the gateway's test mode | Code |
| G7 Differential | Same scenarios produce the same canonical results as the hand-written reference (where one exists) | Comparison harness | Code |
| G8 Human sign-off | Review diff + citations + gate report; signs the package hash | CLI | **Human** |

A package is promoted only if G1–G7 pass and G8 signs **the exact hash** that
was tested. Any change means re-running the gates and getting a new signature.

---

## 8. Runtime security and reliability

1. **Secrets**: loaded from env or keychain through a `SecretStore`. Never in manifests, logs, prompts or the DB in plain text.
2. **Webhooks**: verify against the raw body, compare in constant time, check a timestamp tolerance, protect against replay, store each event once, process idempotently, and apply the forward-only status rule.
3. **Outbound**: TLS only, base URLs allowlisted, strict timeouts, and a circuit breaker per gateway.
4. **Logging**: structured, with card data, keys and signatures redacted. We never store PANs or CVVs; we only use gateway tokens and hosted checkout, which keeps us out of PCI scope.
5. **Reconciliation**: a scheduled job queries each payment that isn't in a final status and fixes any it missed (lost webhooks, timeouts with unknown outcome).
6. **Audit log**: append-only record of who approved which package hash, built by which model and prompts from which doc snapshot.
7. **Kill switch**: disable a gateway or package version at runtime without deploying.

---

## 9. Phases

Each phase has tasks and **exit criteria**. We don't start the next phase until they're met.

### Phase 0 — Core foundations (no external dependencies)
- Repo scaffold: uv, ruff, mypy strict, pytest, Hypothesis, `make check`.
- `Money` + currency table; property tests (no float paths, exponent correctness).
- Canonical models; status state machine; `NextAction`.
- Capability interfaces; `GatewayRegistry` (register/resolve/capabilities; duplicate names rejected).
- Error categories; retry policy rules (unknown outcome is never blindly retried).
- Idempotency service interface + in-memory implementation.
- **Contract suite v1**, parameterised by gateway, + `FakeGateway`.
- **Exit:** `make check` is green; the contract suite passes on FakeGateway; state machine and money have 100% branch coverage.

### Phase 1 — Hand-written references (needs Stripe + Razorpay test keys)
- Hand-written auth and webhook-signature libraries (Basic, Bearer, HMAC variants).
- `StripeGateway`, `RazorpayGateway`; error and status maps.
- Sandbox scenario runner + recorded responses (pytest-recording) for offline replay.
- Webhook ingress with a local tunnel for sandbox webhook delivery.
- **Exit:** both pass G5 + G6 on every scenario they support; the recordings replay offline.

### Phase 2 — Gateway Package engine
- Manifest schema (Pydantic → JSON Schema); `PackageGateway` runtime.
- Mapper sandbox: AST allowlist checker; fixed mapper signatures.
- G1, G2, G4, G7 harnesses; `report.json`; the CLI for signing packages (G8).
- Rewrite Razorpay **by hand** as a package.
- **Exit:** the Razorpay package passes G1–G7 and is **identical in behaviour** to `RazorpayGateway`.

### Phase 3 — Knowledge base and RAG
- Postgres + pgvector schema; doc snapshots; OpenAPI catalogue.
- Crawler + sanitiser + structure-aware chunker; local embeddings; hybrid search.
- Spec Card extraction with citations (Claude).
- **Retrieval evaluation set:** about 50 question→passage pairs for Stripe/Razorpay.
- **Exit:** retrieval recall@5 ≥ 0.9 on the eval set; Stripe/Razorpay spec cards match the hand-written references.

### Phase 4 — Integration agent
- LangGraph workflow (§6.3) with Postgres checkpoints; tools with tiers (§6.4).
- Citation-enforced mapping steps; repair loop; budgets; escalation.
- G3 citation gate.
- **Dev evaluation:** regenerate Razorpay and Stripe from docs alone and compare to the references.
- **Holdout evaluation:** PayPal + Cashfree, never seen during development.
- **Exit:** the dev gateways pass G1–G7; on the holdout gateways we report gate pass rates honestly; nothing is promoted without G8.

### Phase 5 — Runtime service
- FastAPI payments API; Postgres ledger + idempotency store; webhook ingress.
- Reconciler; circuit breaker; kill switch; audit log; OpenTelemetry + structlog.
- **Exit:** end-to-end sandbox payment → webhook → reconciled final status, through a package the AI generated and a human approved.

### Phase 6 — Ecosystem
- MCP server: `search_gateway_docs`, `get_endpoint_schema`, `run_contract_tests`, `get_package_report` (read and sandbox only).
- Doc change detection: re-snapshot on a schedule, diff, and flag affected package fields → regenerate → gates → sign.
- Routing strategies (failover, preference by capability).
- Optional: a web review UI.

---

## 10. Evaluation metrics (for the README and interviews)

- **Gate pass rate** per holdout gateway (G1–G7), first try vs after repair.
- **Field accuracy** vs the hand-written reference (endpoints, field maps, status/error maps).
- **Citation precision**: the share of citations that actually support the claim.
- **Repair loops** needed, tokens and cost per integration, wall-clock time.
- **Human review time** per package.

---

## 11. Repository layout

```
payweave/
  src/payweave/
    core/          money, models, states, errors, capabilities, registry, idempotency
    gateways/      stripe/, razorpay/ (hand-written references)
    package/       manifest schema, PackageGateway, mapper sandbox, auth & signature libs
    knowledge/     snapshots, openapi catalogue, crawler, chunker, embedder, retriever
    agent/         langgraph workflow, nodes, tools, prompts
    verify/        gates G1–G7, scenario runner, differential harness, reports
    runtime/       FastAPI app, services, webhook ingress, reconciler, ledger
    mcp/           MCP server (Phase 6)
    cli.py         ingest / generate / verify / sign / register
  tests/           unit, property, contract/, scenarios/
  packages/        signed gateway packages
  docs/            ADRs
```

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sandbox doesn't match production behaviour | Recordings from the real sandbox, reconciliation at runtime, a gradual rollout behind a kill switch |
| Gateway has no OpenAPI spec | G4 weakens, so more weight goes on G6/G7 and stricter citation checks; the report shows the reduced coverage |
| Docs are wrong or contradict each other | Sandbox results win over docs; contradictions are flagged for a human |
| Agent loops or burns tokens | Hard budgets, escalation with a diagnosis |
| Injection via docs | §6.5: permissions enforced in code, no secrets in context, domain allowlist |
| Gateway needs an auth/signature scheme we don't support | The agent stops and a human adds it to the hand-written library |

## 13. What you'll need, and when

| When | What |
|---|---|
| Phase 1 | Razorpay test account ✅ (keys in `.env`); Cashfree sandbox account |
| Phase 1 | A tunnel for webhooks: `zrok` (Razorpay blacklists ngrok/localtunnel) |
| Phase 3 | Postgres 18 + pgvector (already installed); Claude API key only for contextual retrieval |
| Phase 4 | PayPal developer sandbox (holdout) |
