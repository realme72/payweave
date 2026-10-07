from pathlib import Path
from typing import Any

import pytest

from payweave.knowledge.chunker import ChunkingConfig, ChunkKind, ParsedDocument, chunk_markdown


def words(text: str) -> int:  # stand-in tokenizer: one token per word
    return len(text.split())


API_PAGE = """---
title: Capture a Payment
description: Capture it.
---

# Capture a Payment

**POST** `/v1/payments/:id/capture`

Use this endpoint to capture a payment.

### Request

```curl: Curl
curl -X POST https://api.razorpay.com/v1/payments/pay_1/capture

```java: Java
RazorpayClient razorpay = new RazorpayClient("key", "secret");

```python: Python
import razorpay
```

### Parameters

`id` _mandatory_
: `string` The payment id.

### Parameters

`amount` _mandatory_
: `integer` Amount to capture.

`currency` _mandatory_
: `string` Currency \\{code\\}.

### Parameters

`id`
: `string` Unique identifier of the payment.

`status`
: `string` Payment status.

### Errors

Capture amount must be equal to the amount authorized.
* code: 400
* description: Wrong amount.
* solution: Use the authorized amount.

Payment is not in authorized state.
* code: 400
* description: Not authorized.
* solution: Wait for authorization.
"""


def parse(text: str = API_PAGE, **config: Any) -> ParsedDocument:
    return chunk_markdown(
        text, gateway="razorpay", count_tokens=words, config=ChunkingConfig(**config)
    )


def by_kind(doc: ParsedDocument, kind: ChunkKind) -> list[str]:
    return [c.body for c in doc.chunks if c.kind is kind]


def test_metadata_and_endpoint() -> None:
    doc = parse()
    assert doc.title == "Capture a Payment"
    assert doc.endpoint == "POST /v1/payments/:id/capture"


def test_three_identical_parameter_headings_are_told_apart() -> None:
    doc = parse()
    assert by_kind(doc, ChunkKind.PATH_PARAMS) == ["`id` _mandatory_\n: `string` The payment id."]
    [body] = by_kind(doc, ChunkKind.REQUEST_PARAMS)
    assert "`amount` _mandatory_" in body
    assert "`currency` _mandatory_" in body
    [response] = by_kind(doc, ChunkKind.RESPONSE_FIELDS)
    assert "`status`" in response
    headers = {c.kind: c.header for c in doc.chunks}
    assert headers[ChunkKind.REQUEST_PARAMS] == (
        "Razorpay > Capture a Payment > Request body parameters\nPOST /v1/payments/:id/capture"
    )


def test_sdk_samples_dropped_curl_kept_even_with_unclosed_tabs() -> None:
    text = "\n".join(c.text for c in parse().chunks)
    assert "curl -X POST" in text
    assert "RazorpayClient" not in text
    assert "import razorpay" not in text
    keep_all = "\n".join(c.text for c in parse(drop_sdk_samples=False).chunks)
    assert "RazorpayClient" in keep_all


def test_each_error_is_its_own_chunk() -> None:
    errors = by_kind(parse(), ChunkKind.ERROR)
    assert len(errors) == 2
    assert errors[0].startswith("Capture amount must be equal")
    assert errors[1].startswith("Payment is not in authorized state")


def test_escaped_braces_are_unescaped() -> None:
    assert "Currency {code}." in by_kind(parse(), ChunkKind.REQUEST_PARAMS)[0]


def test_headings_inside_code_blocks_are_not_sections() -> None:
    doc = parse("# T\n\n```bash: CLI\n# not a heading\necho hi\n```\n")
    assert [s.heading_path for s in doc.sections] == [("T",)]
    assert "# not a heading" in doc.chunks[0].body


def test_table_rows_become_labelled_standalone_chunks() -> None:
    page = (
        "# List of Errors\n\n## Bad Request\n\n"
        + "\n---\n".join(
            [
                'Error Reason [columWidth="25"] | Description | Next Steps',
                "card_expired | The card has expired. | Retry with a valid card.",
                "payment_failed | Bank error. | Retry later.",
            ]
        )
        + "\n---\n"
    )
    rows = [c.body for c in parse(page).chunks]
    assert rows == [
        "Error Reason: card_expired\nDescription: The card has expired.\n"
        "Next Steps: Retry with a valid card.",
        "Error Reason: payment_failed\nDescription: Bank error.\nNext Steps: Retry later.",
    ]


def test_size_limits_hold_and_small_prose_merges() -> None:
    long_paragraph = " ".join(f"word{i}." for i in range(400))
    doc = parse(f"# T\n\n{long_paragraph}\n\nTiny tail.", max_tokens=120, target_tokens=80)
    assert all(c.token_count <= 120 for c in doc.chunks)
    assert len(doc.chunks) > 3
    assert "Tiny tail." in doc.chunks[-1].body  # merged, not a 2-word chunk of its own
    assert all(len(c.body.split()) >= 2 for c in doc.chunks)


def test_content_hash_tracks_the_source() -> None:
    assert parse().content_hash == parse().content_hash
    assert parse().content_hash != parse(API_PAGE + "\nmore").content_hash


SNAPSHOT = Path(__file__).resolve().parents[2] / "docs" / "sources" / "razorpay"


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="doc snapshot not fetched (make docs-fetch)")
def test_real_snapshot_never_exceeds_the_model_limit() -> None:
    # Words underestimate model tokens, so check with a deliberately pessimistic counter.
    def pessimistic(text: str) -> int:
        return len(text) // 3

    for path in SNAPSHOT.glob("*.md"):
        doc = chunk_markdown(path.read_text(), gateway="razorpay", count_tokens=pessimistic)
        assert all(c.token_count <= 512 for c in doc.chunks), path.name
        assert doc.chunks, path.name
