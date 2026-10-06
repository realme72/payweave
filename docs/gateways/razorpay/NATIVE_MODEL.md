# Razorpay: native model (from docs)

Every fact cites a snapshot in `docs/sources/razorpay/` (hashes and fetch times
in `SNAPSHOT.tsv`). **[VERIFY]** marks things the docs don't state clearly; we
confirm them in the sandbox before relying on them.

---

## 1. Entities

| Razorpay entity | What it is | Our concept |
|---|---|---|
| **Order** | Created by our server with amount + currency. The customer pays against it in Checkout. Can have several payment attempts. | The **payment intent**: what the merchant wants to collect |
| **Payment** | One attempt by the customer (card, UPI, netbanking...). Created by Razorpay when the customer submits details. | A **payment attempt** |
| **Refund** | Refund against a captured payment | Refund |

Key point: **our server never creates a Razorpay payment.** It creates an
*order*, and the customer creates payments in Razorpay Checkout.
(`api__payments__entity.md`, `payments__payment-gateway__web-integration__standard__integration-steps.md`)

---

## 2. Native state machines

### 2.1 Order (`api__orders__entity.md`)
```
created ──(first payment attempted)──► attempted ──(a payment captured)──► paid
```
- "`paid`... No further payment requests are permitted... The order stays in the
  `paid` state even if the payment associated with the order is refunded."

### 2.2 Payment (`payments__payments.md`, `payments__payments__late-authorisation.md`)
```
           ┌──────────────► failed ──(late authorisation)──┐
           │                                               ▼
created ───┴──────────────────────────────────────────► authorized ──► captured ──► refunded
                                                           │
                                                           └──(not captured in time: auto-refund)──► refunded
```
| Transition | Source |
|---|---|
| created → authorized | "changes to `authorized` when the bank successfully authenticates" |
| created → failed | "unsuccessful payment attempt is marked as `failed`"; also no bank response for 10 min → "marked as `Failed` due to timeout" |
| **failed → authorized** | **Late authorisation**: "If Razorpay receives the payment status from the bank later, payment is moved to the `authorized` state" (Razorpay polls the bank for 3 days) |
| authorized → captured | manual capture API or auto-capture settings |
| authorized → refunded | "auto-refunded to the customer if not captured within 3 days" (`payments__payments.md`). **Inconsistent:** other pages say 5 days (`capture-settings.md`, `late-authorisation__handle.md`). We don't depend on the duration. |
| captured → refunded | merchant refund |

Refund fields on the payment (`api__payments__entity.md`): `refund_status` ∈ {`null`, `partial`, `full`}, `amount_refunded`, `captured` (bool).
- **[VERIFY]** After a *partial* refund, does `status` stay `captured` with `refund_status=partial`? The docs imply this but don't say it outright.

### 2.3 Refund (`api__refunds__entity.md`)
```
pending ──► processed   (final)
   └──────► failed      (e.g. normal refund on a payment > 6 months old)
```

---

## 3. Webhooks

| Event | Object in payload | Source |
|---|---|---|
| `payment.authorized` | payment | `webhooks__payments.md` |
| `payment.captured` | payment | same |
| `payment.failed` | payment | same |
| `order.paid` | order + payment | `webhooks__orders.md` |
| `refund.created` / `refund.processed` / `refund.failed` | refund (+ payment) | `webhooks__refunds.md` |
| `refund.speed_changed` | refund | same, ignored by us |

Rules from the docs:
- **Signature:** header `X-Razorpay-Signature` = HMAC-SHA256 of the **raw body** with the **webhook secret** (not the API key secret). "Do not parse or cast the webhook request body." (`webhooks__validate-test.md`). **[VERIFY]** hex encoding (the official SDKs use a hex digest).
- **Duplicates happen**: dedupe on the `x-razorpay-event-id` header.
- **Order isn't guaranteed**: "`payment.authorized`, `payment.captured`... may not be followed at all times."
- **The payload is a snapshot at event time**: "The payment can be in the `captured` state when the `payment.authorized` Webhook is fired." → **We treat a webhook as a hint and re-fetch the payment from the API before updating state.**
- **`payment.failed` isn't always sent**: "not triggered if the payment fails during authorisation (while making the first payment)" → **reconciliation by polling is required, not optional.**
- Secret rotation: old events are still signed with the *old* secret, so we must support validating against more than one secret.
- Local testing: `ngrok.io`, `localhost`, `loca.lt` and others are **blacklisted**; Razorpay suggests `zrok`.

## 4. Checkout signature (client → our server)

After Checkout succeeds the browser sends us `razorpay_payment_id`, `razorpay_order_id` and `razorpay_signature`, where
`signature = hex(HMAC-SHA256(order_id + "|" + payment_id, key_secret))` (`integration-steps.md`).
We must verify it **and** still re-fetch the payment status from the API. A valid signature proves who sent it, not the payment's final state.

## 5. Operations → capabilities

| Operation | Razorpay | Notes |
|---|---|---|
| Create intent | `POST /v1/orders` | `receipt` ≤ 40 chars, "has to be unique"; `notes` ≤ 15 pairs × 256 chars |
| Capture | `POST /v1/payments/:id/capture` | **Full amount only**: "Capture amount must be equal to the amount authorized"; `amount` + `currency` both required |
| Void | **None** | Uncaptured payments are auto-refunded on expiry |
| Refund | `POST /v1/payments/:id/refund` | Partial ✓, multiple ✓; **native idempotency** via `X-Refund-Idempotency` (≥ 10 chars, `[A-Za-z0-9_-]` only; same body required) |
| Fetch | `GET /v1/orders/:id`, `/v1/orders/:id/payments`, `/v1/payments/:id`, `/v1/refunds/:id` | |
| Capture mode | Order `payment.capture` = `automatic`/`manual` + expiry periods | Overrides dashboard settings |

→ Capabilities: CREATE_PAYMENT, CAPTURE, MANUAL_CAPTURE, REFUND, PARTIAL_REFUND,
MULTIPLE_REFUNDS, GET_PAYMENT, GET_REFUND, WEBHOOKS. **Not supported:** VOID, PARTIAL_CAPTURE.
Native idempotency applies to **refunds only**.

## 6. Money rules
- Amounts are in the smallest unit (matches our `Money`).
- **3-decimal currencies (KWD, BHD, OMR): the last digit must be 0**: "pass... `99990` and not `99991`". This is a Razorpay-specific check that belongs in its mapper.

## 7. Errors (`errors.md`, `errors__payments__list.md`)
Error body: `code` (e.g. `BAD_REQUEST_ERROR`, `GATEWAY_ERROR`, `SERVER_ERROR`), `description`,
`source` (customer/business/gateway/razorpay), `step`, `reason` (machine-readable, e.g.
`insufficient_funds`, `incorrect_otp`, `card_expired`, `order_already_paid`). We map on
**`reason`** first, then `code`. The full table is built in Phase 1.

---

## 8. Conflicts with our canonical model → decisions (all accepted 2026-10-07, implemented)

| # | Finding | Proposed decision |
|---|---|---|
| **D1** | Razorpay separates the **intent** (order) from **attempts** (payments). One order can have many attempts. Stripe does the same (PaymentIntent + charges). Our canonical `Payment` mixes the two. | Split canonical into **`PaymentIntent`** (what merchants see; one per merchant order) and **`PaymentAttempt`** (each try). Intent status is *derived* from its attempts. |
| **D2** | **Late authorisation: failed → authorized.** Our canonical FAILED is terminal. | Solved by D1: a failed *attempt* doesn't fail the *intent* (the customer can retry, and a late auth can revive it). The attempt machine allows `FAILED → AUTHORIZED`, and that's still acyclic. |
| **D3** | Uncaptured payments end as native `refunded` though nothing was ever captured. | Map native `refunded` + `captured=false` → canonical **VOIDED**. This is a mapper function, not a lookup table. |
| **D4** | Our server can't complete the payment; the customer must open Razorpay Checkout. | New `NextAction`: **`client_checkout`**, carrying only public data (`key_id`, `order_id`). Never the secret. |
| **D5** | Razorpay refund idempotency keys must be ≥ 10 chars of `[A-Za-z0-9_-]`. Ours allow 8+ chars plus `:` and `.`. | Tighten the canonical key to **16–255 chars of `[A-Za-z0-9_-]`**, which is safe for every gateway seen so far. |
| **D6** | Native idempotency exists only for refunds. | Make `NATIVE_IDEMPOTENCY` per operation (e.g. `NATIVE_IDEMPOTENT_REFUND`); our idempotency store covers everything else. |
| **D7** | Orders never "fail" natively; they stay `created`/`attempted`. | Intent-level `FAILED` is not used by Razorpay. An intent with no successful attempt becomes **EXPIRED** by our own timeout policy (new status), and attempts carry the failure details. |

## 9. To verify in the sandbox (Phase 1)
- V1: status after a partial refund (§2.2)
- V2: webhook signature encoding is hex (§3)
- V3: auto-refund timing for uncaptured payments in test mode (3 vs 5 days)
- V4: the `payment.authorized` sample with `captured: true`: is that a docs mistake or real behaviour?
- V5: test-mode outcomes we can trigger (success, failure, insufficient funds, OTP) using Razorpay's test cards and UPI IDs
