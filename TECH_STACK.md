# PayWeave — Tech Stack & Rationale

What we use, why we picked it over the alternatives, and what we gain from it.

The whole stack follows one rule: **AI builds, deterministic code runs and checks.**
Anything that touches money or decides whether something is correct is ordinary,
testable code. The AI's job is to read docs and propose a manifest, and nothing
it proposes is trusted until checks that don't use AI have passed.

---

## 1. Core platform (the deterministic side)

| Tech | Role | Why this one | What it gives us |
|---|---|---|---|
| **Python 3.13** | Main language | Strongest AI/RAG ecosystem; you already know it from Sentinel-AI | One language from the AI pipeline to the payment runtime |
| **FastAPI** | HTTP API (payments, webhooks, review endpoints) | Async, built on Pydantic, generates OpenAPI docs automatically | Fast webhook handling and typed request/response contracts |
| **LangGraph** | Agent workflow graph | Stateful graph with loops, human-in-the-loop interrupts, Postgres checkpoints | Resumable, replayable, auditable agent runs |
| **Pydantic v2** | Canonical models + **Manifest schema** | Strict validation with a Rust core; exports JSON Schema | The *same* schema validates AI output, constrains Claude's structured output and documents the manifest format, so there's one source of truth |
| **httpx** | Outbound calls to gateways | Async, timeouts, pluggable transports | Easy to mock in tests and gives us retry/timeout control |
| **PostgreSQL 18** | Payments, idempotency keys, audit log, manifests | ACID transactions, which money needs; already installed natively on your Mac | One database for state, RAG and audit. No Redis or Mongo to run |
| **SQLAlchemy 2 + Alembic** | ORM + migrations | The standard choice, async-capable | Schema changes tracked and versioned |
| **uv** | Packaging / env | Much faster than pip/poetry, and lockfiles are reproducible | Quick setup, same dependencies on every machine |

### Design patterns → where they live

- **Strategy:** `PaymentGateway` interface. Each gateway (hand-written or manifest-driven) is a strategy, and routing strategies (cheapest, failover) follow the same pattern.
- **Registry:** `GatewayRegistry` resolves strategies by name. Adding a gateway means *registering* it, never editing core code (Open/Closed).
- **Adapter:** each gateway converts its own wire format to and from our canonical models.
- **Factory:** builds a gateway instance from a manifest plus credentials.
- **Template Method:** shared request pipeline (auth → build request → send → map response → map errors), with gateway-specific steps.
- **Dependency Inversion:** core depends on interfaces (`LLMProvider`, `Embedder`, `VectorStore`, `SecretStore`), never on concrete vendors.

---

## 2. AI technologies

### 2.1 LLM — Claude API (Claude Opus 5.5)

**Why Claude:**
- **1M-token context:** a whole gateway's API reference fits in one request, so the model sees the full picture and not just fragments.
- **Structured outputs** (`output_config.format` / `client.messages.parse()`): the manifest is generated *against our Pydantic JSON Schema*, so badly formed output can't happen and only the semantics need checking.
- **Strict tool use** (`strict: true`): the agent's tool calls always match their schemas.
- **Citations:** answers can point at the exact passage in the source document. That gives us provenance for every mapping.
- **Prompt caching:** a gateway's docs are reused across many agent turns. Cached reads cost about 10% of normal input, so repeated turns over the same docs get much cheaper.
- **Adaptive thinking + effort control:** use high effort for mapping and planning and lower effort for routine steps.
- **Batch API (50% cheaper):** for running evaluation suites across many gateways overnight.

**Why not the alternatives:**
- *Groq free tier:* 200k tokens/day isn't enough to read one large gateway's docs a few times over.
- *Local models (Ollama):* free, but noticeably weaker at long-document reasoning and following schemas, which is exactly where we need accuracy.
- The LLM sits behind an `LLMProvider` interface, so this choice can be swapped later.

**Model choice:** Claude Opus 5.5 (`claude-opus-5-5`) is the default because accuracy is the requirement. We'll only consider a cheaper model for simple steps if evaluations show quality holds.

### 2.2 RAG (Retrieval-Augmented Generation)

**What it is:** we store the gateway docs as searchable chunks and give the model only the relevant, cited pieces for each question.

**Why we need it even with 1M context:**
- **Precision:** "What is Razorpay's webhook signature algorithm?" should be answered from *that exact section*, with a citation, not from the model's memory of something it may have seen in training.
- **Provenance:** every manifest field stores the chunk ID it came from. A reviewer can click through to the source, and the verifier can check the claim against it.
- **Cost:** small targeted contexts per step, instead of resending everything.
- **Change detection:** when the docs change, re-ingest them and see exactly which manifest fields are affected.

**How we build it:**

| Piece | Choice | Why |
|---|---|---|
| Spec parsing | **OpenAPI parser (deterministic)** | If a gateway publishes an OpenAPI spec, endpoints and field types come from parsing, not AI. That's 100% accurate for those facts |
| Doc ingestion | HTML/Markdown/PDF crawler | Many gateways document webhooks, signatures and status flows only in prose |
| Chunking | **Structure-aware**: one chunk per API operation, prose split by heading | Keeps "endpoint + its parameters + its example" together |
| Embeddings | **Local open-source model** (e.g. BGE via `fastembed`) behind an `Embedder` interface | Free, runs offline and docs never leave your machine. Anthropic doesn't offer an embeddings API, and Voyage AI is an optional paid upgrade |
| Vector store | **pgvector** in the same Postgres | No extra service, and vector search can be joined with relational data |
| Search | **Hybrid: vector + Postgres full-text (BM25-style)** | API docs are full of exact identifiers (`amount_capturable`, `X-Razorpay-Signature`). Vector search alone misses exact matches and keyword search alone misses paraphrases, so we combine both |

### 2.3 Agentic AI

**What it is:** instead of one big prompt, an agent works in steps, uses tools, checks results and fixes its own mistakes.

**Why an agent and not a single call:** building an integration really is multi-step: understand auth → map each operation → map statuses → work out webhooks → test → fix. Each step's errors can be caught by a tool (tests and the validator), which is exactly when agents are worth the extra cost.

**Workflow (LangGraph graph, see PLAN.md §6.3):**
understand provider → Spec Card → [human confirms] → plan capabilities → map operations →
map status & errors → map webhooks → write Gateway Package → run gates → (fail → diagnose → repair, max N) →
sandbox → differential check → security review → [human approves] → human registers.

**What the AI writes:** a **Gateway Package**: a declarative manifest plus small, pure, typed mapper
functions, each citing its source. HTTP, auth, signature crypto, retries and idempotency are
hand-written libraries the manifest *selects* from. The AI never writes them.

**Agent tools, by permission tier:**
- *Read:* `search_docs`, `get_endpoint_schema`, `get_spec_card`, `get_reference_package`
- *Write (sandboxed):* `write_package_file` (run's working directory only)
- *Execute (sandbox keys only):* `validate_manifest`, `run_static_checks`, `run_contract_tests`, `run_sandbox_scenario`
- *Never an agent tool:* `register_gateway`, anything with live keys. Those are human-only.

**How we build it: LangGraph + the Anthropic SDK directly.**
- **Why LangGraph:** the workflow really is a graph with a conditional repair loop and two
  human-approval pauses. LangGraph gives us interrupts for human approval, Postgres
  checkpoints (resume or replay any run) and explicit state, so we don't hand-build them.
- **Why not LangChain's model wrappers:** inside each step we call Claude through the
  official `anthropic` SDK, which gives us full control over structured outputs, citations,
  prompt caching and thinking/effort, with no hidden prompt rewriting.
- **Why not Managed Agents:** contract tests and sandbox calls must run *on our machine with
  our test keys*. Self-hosting keeps secrets local.
- **Prompt-injection defence:** crawled docs are untrusted data. Tool permissions and domain
  allowlists are enforced in code, so a successful injection still can't do anything dangerous.

### 2.4 MCP (Model Context Protocol), Phase 6

An MCP server exposes `search_gateway_docs`, `get_endpoint_schema`, `run_contract_tests` and
`get_package_report` (read and sandbox only), so PayWeave can be used from Claude Code or any
MCP-capable agent. It never exposes registration or live-key operations.

---

## 3. Verification & security tooling

| Tech | Role | Benefit |
|---|---|---|
| **pytest** | Contract test suite every gateway must pass | One definition of "correct", applied to hand-written and AI-generated gateways alike |
| **Hypothesis** | Property-based tests (money math, status transitions) | Finds edge cases (zero amounts, 3-decimal currencies like KWD, rounding) that we'd never write by hand |
| **respx** | Mock httpx at the transport level | Fast offline tests with no network |
| **pytest-recording (VCR)** | Record real sandbox responses, replay offline | Contract tests run against *real* gateway behaviour without network or keys |
| **Differential harness** (ours) | Run the same scenarios on an AI-generated package and on the hand-written reference, then compare canonical results | The strongest correctness signal: two independent implementations agreeing |
| **AST allowlist checker** (ours) | Mapper functions may only use approved imports/calls; no I/O, eval or globals | Keeps AI-written code pure and safe to run |
| **zrok** tunnel | Deliver sandbox webhooks to localhost (Razorpay blacklists ngrok/localtunnel) | Real webhook + signature testing during development |
| **Schemathesis** | Fuzzes requests against the gateway's OpenAPI spec | Catches manifest requests that don't match the real API |
| **Gateway sandboxes** (Stripe, Razorpay, PayPal, Cashfree test modes) | Real end-to-end runs | The final proof against the real API, all free |
| **mypy (strict) + ruff** | Type checking + linting | Catches whole classes of bugs before runtime |
| **bandit + semgrep** | Static security scan, especially of code hooks | Blocks dangerous patterns (eval, shell, unexpected network calls) |
| **cryptography / hmac** (stdlib) | Webhook signatures, constant-time compare | Correct, audited primitives. We never write our own crypto |

---

## 4. Observability

| Tech | Benefit |
|---|---|
| **structlog** | Structured JSON logs with secrets automatically redacted |
| **OpenTelemetry** | Traces across API → gateway → webhook, plus traces of each agent step |
| **Audit table** | Who approved which manifest version, generated by which model from which doc snapshot |

---

## 5. What we deliberately *don't* use

| Skipped | Why |
|---|---|
| LLM in the payment path | Not deterministic, and we can't audit it at transaction time |
| LangChain model/chain wrappers | We call Claude via the official SDK for full control; LangGraph is used only for the workflow graph |
| Separate vector DB (Qdrant/Pinecone) | pgvector is enough at our scale, so one fewer service |
| Redis | Postgres handles idempotency keys transactionally |
| Docker (for now) | Not available on this Mac; everything runs natively. We can add it later for deployment |
| Floats for money | Integer minor units only |

---

## 6. Summary of benefits

- **Correctness:** the OpenAPI is parsed deterministically, outputs are schema-constrained, every field cites its source, and contract tests plus sandbox runs and human approval sit in front of anything going live.
- **Security:** no LLM at runtime, secrets kept out of prompts and logs, constant-time signature checks, static scans of hooks, a full audit trail.
- **Extensibility:** Strategy + Registry means a new gateway is just a new manifest, with no core changes.
- **Cost:** free local embeddings, free sandboxes, prompt caching and batch evaluations keep Claude spend low.
- **Simplicity:** one language, one database, no Docker; it all runs on your Mac.
