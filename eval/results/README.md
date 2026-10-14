# Holdout results

Spec Cards the agent extracted, kept so progress can be compared over time. Each is
scored against the answer key in `eval/spec_cards/`, written before any agent run.

## cashfree-spec-card.json (2026-10-14)

Merged from per-section runs on Gemini Flash (basics, refunds, capture, api_keys,
webhooks, errors) and Groq qwen3.8-27b (payment_flow, endpoints, ids, error_shape,
checkout_return), each section's facts re-checked against its citations.

**16/28 correct, 2 wrong, 10 abstained.** Reproduce with
`uv run python -m payweave.agent score eval/results/cashfree-spec-card.json`.

- Wrong: `errors.error_code_field = code`, `errors.reason_field = message`.
- Not in the key, checked by hand: `api_keys` and `checkout_return` all null (correct:
  the docs show no key prefixes and nothing is signed on return); `ids` are real samples,
  but `cf_order_id` is Cashfree's id, while the paths take the merchant's `order_id`.
