.PHONY: check lint typecheck test fmt docs-fetch secrets-scan jaeger trace-demo kb-ingest kb-eval

check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

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
