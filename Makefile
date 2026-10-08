.PHONY: gates packages-verify check lint typecheck test fmt docs-fetch secrets-scan jaeger trace-demo kb-ingest kb-eval

check: lint typecheck test packages-verify

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy
	@# Gate G1: each Gateway Package's mappers, strict, one at a time (same module name)
	for f in packages/*/*/mappers.py; do uv run mypy --strict "$$f" || exit 1; done

test:
	uv run pytest --cov --cov-report=term-missing

fmt:
	uv run ruff format .
	uv run ruff check --fix .

docs-fetch:
	./scripts/fetch_docs.sh

secrets-scan:
	gitleaks dir . --redact --no-banner

jaeger:
	./scripts/jaeger.sh

# Needs `make jaeger` running and test keys in .env
trace-demo:
	set -a; . ./.env; set +a; uv run --extra otel python scripts/trace_demo.py

# Knowledge base (needs local Postgres + pgvector; database "payweave")
kb-ingest:
	uv run --extra knowledge python -m payweave.knowledge ingest razorpay

kb-eval:
	uv run --extra knowledge python -m payweave.knowledge eval eval/retrieval/razorpay.toml

# Gates G1-G7 on a package; rewrites its report.json (PKG=packages/<gateway>/<version>)
PKG ?= packages/razorpay/v1
gates:
	uv run python -m payweave.verify $(PKG)

# Gate G8: every Gateway Package must carry a valid approval from a trusted signer
# and a current, promotable report.json
packages-verify:
	uv run python -m payweave.packages verify
