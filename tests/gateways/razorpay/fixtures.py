"""Razorpay payloads shaped like the samples in docs/sources/razorpay/.

Sources: api__orders__entity.md, api__payments__entity.md (success + failure
samples), api__refunds__entity.md, webhooks__payments.md.
"""

from typing import Any

ORDER_ID = "order_DaZlswtdcn9UNV"
KEY_ID = "rzp_test_abcdefghijklmn"


def order(status: str = "created", amount: int = 50000, **kw: Any) -> dict[str, Any]:
    return {
        "id": ORDER_ID,
        "entity": "order",
        "amount": amount,
        "amount_paid": 0,
        "amount_due": amount,
        "currency": "INR",
        "receipt": "Receipt #20",
        "status": status,
        "attempts": 0,
        "notes": {"pw_capture_mode": "manual", "pw_idempotency_key": "k" * 16, "cart": "42"},
        "created_at": 1572502745,
        **kw,
    }


def payment(
    status: str,
    *,
    pid: str = "pay_L0nSsccovt6zyp",
    amount: int = 50000,
    refunded: int = 0,
    captured: bool | None = None,
    created_at: int = 1572502800,
    **kw: Any,
) -> dict[str, Any]:
    failed = status == "failed"
    return {
        "id": pid,
        "entity": "payment",
        "amount": amount,
        "currency": "INR",
        "status": status,
        "order_id": ORDER_ID,
        "method": "card",
        "amount_refunded": refunded,
        "refund_status": None if refunded == 0 else ("full" if refunded == amount else "partial"),
        "captured": captured if captured is not None else status in ("captured", "refunded"),
        "notes": [],  # Razorpay sends [] for empty notes in its samples
        "error_code": "BAD_REQUEST_ERROR" if failed else None,
        "error_description": "Your payment has been cancelled." if failed else None,
        "error_source": "customer" if failed else None,
        "error_step": "payment_authentication" if failed else None,
        "error_reason": "payment_cancelled" if failed else None,
        "created_at": created_at,
        **kw,
    }


def payments(*items: dict[str, Any]) -> dict[str, Any]:
    return {"entity": "collection", "count": len(items), "items": list(items)}


def refund(status: str = "processed", amount: int = 10000, **kw: Any) -> dict[str, Any]:
    return {
        "id": "rfnd_FgRAHdNOM4ZVbO",
        "entity": "refund",
        "amount": amount,
        "currency": "INR",
        "payment_id": "pay_L0nSsccovt6zyp",
        "notes": {},
        "status": status,
        "speed_requested": "normal",
        "created_at": 1600856650,
        **kw,
    }


def webhook(event: str, **payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": "event",
        "account_id": "acc_BFQ7uQEaa7j2z7",
        "event": event,
        "contains": list(payload),
        "payload": {name: {"entity": entity} for name, entity in payload.items()},
        "created_at": 1567674606,
    }
