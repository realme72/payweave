# Holdout results

Spec Cards the agent extracted, kept so progress can be compared over time. Each is
scored against the answer key in `eval/spec_cards/`, written before any agent run.

## cashfree-spec-card.json (2026-10-14)

Merged from per-section runs on Gemini Flash (basics, refunds, api_keys, webhooks,
errors) and Groq qwen3.8-27b (payment_flow, ids, checkout_return; endpoints,
error_shape and capture after the fixes below), each section's facts re-checked
against its citations.

**22/28 correct, 3 wrong, 3 abstained.** Reproduce with
`uv run python -m payweave.agent score eval/results/cashfree-spec-card.json`.

| Step | Score |
| --- | --- |
| First complete card | 16 correct, 2 wrong, 10 abstained |
| + endpoint index (one citable passage per API page) for endpoints and capture | endpoints 2/6 → 6/6, void found |
| + OpenAPI YAML examples accepted as sample bodies (`message:` as well as `"message":`) | error_shape message found |
| **Now** | **22 correct, 3 wrong, 3 abstained** |

Prompts were not tuned on the wrong answers below; that would fit the holdout.

- Wrong: `errors.error_code_field = code`, `errors.reason_field = message`, and
  `error_shape.code_path = code` (its quote, `code: order_already_exists`, didn't show
  `type`, which carries the broad code).
- Not in the key, checked by hand: `api_keys` and `checkout_return` all null (correct:
  the docs show no key prefixes and nothing is signed on return); `ids` are real samples,
  but `cf_order_id` is Cashfree's id, while the paths take the merchant's `order_id`.
