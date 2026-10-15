# Holdout results

Spec Cards the agent extracted, kept so progress can be compared over time. Each is
scored against the answer key in `eval/spec_cards/`, written before any agent run.

## cashfree-spec-card.json (2026-10-14)

Merged from per-section runs on Gemini Flash (basics, refunds, api_keys, webhooks,
errors) and Groq qwen3.8-27b (payment_flow, ids, checkout_return; endpoints,
error_shape and capture after the fixes below), each section's facts re-checked
against its citations.

**34/35 correct, 0 wrong, 1 abstained** (`webhooks.ordered_delivery`). Reproduce with
`uv run python -m payweave.agent score eval/results/cashfree-spec-card.json`.

| Step | Score |
| --- | --- |
| First complete card | 16 correct, 2 wrong, 10 abstained (of 28) |
| + endpoint index for endpoints and capture | endpoints 2/6 → 6/6, void found |
| + OpenAPI YAML examples accepted as sample bodies | error_shape message found |
| + new fields, answers written before their runs: auth_headers (4), sandbox_base_url, webhooks timestamp_header and signature_encoding; `api_key_headers` added as an auth choice | 30/35, 3 wrong |
| + **sandbox error probe** (verify/error_probe.py): three harmless failing requests; `type` stayed `invalid_request_error`, `code` changed every time | errors + error_shape: 3 wrong and 1 abstained → all correct |

The four error fields are cited `observed:sandbox`: evidence from Cashfree's own API, not
a reading of its docs. Without it the docs-only score is 30/35 with 3 wrong.

Prompts were not tuned on the wrong answers; the sandbox settled them instead.
- Not in the key, checked by hand: `api_keys` and `checkout_return` all null (correct:
  the docs show no key prefixes and nothing is signed on return); `ids` are real samples,
  but `cf_order_id` is Cashfree's id, while the paths take the merchant's `order_id`.
