"""Compile a Spec Card into a Gateway Package manifest, deterministically.

The model never writes the manifest. It extracts cited facts (the card); this
module turns them into ``manifest.toml`` with fixed rules: sample ids become
anchored id patterns, documented endpoints become paths relative to the base
URL, key prefixes become the live-key guard. Each manifest value records the
citations it came from (``provenance``), and anything the card leaves open is
reported as a gap for a person instead of being guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import tomli_w
from pydantic import ValidationError

from payweave.agent.spec_card import Cited, SpecCard
from payweave.packages.manifest import Manifest
from payweave.verify.error_probe import ErrorShapeObserved

# Our credential names per auth scheme (PayWeave's choice, not the gateway's).
# api_key_headers uses the basic names: a key id and its secret.
BASIC_USER, BASIC_SECRET, BEARER_SECRET = "key_id", "key_secret", "secret_key"
WEBHOOK_SECRETS = "webhook_secrets"
_PLACEHOLDER = re.compile(r":[A-Za-z_]\w*|\{[A-Za-z_]\w*\}")


@dataclass
class Compiled:
    manifest: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)  # needs a person; the manifest is incomplete
    notes: list[str] = field(default_factory=list)  # deliberate omissions, for the reviewer
    problems: list[str] = field(default_factory=list)  # the result failed Manifest validation

    @property
    def ok(self) -> bool:
        return not self.gaps and not self.problems

    def toml(self) -> str:
        return tomli_w.dumps(self.manifest)


class _Builder:
    def __init__(self, card: SpecCard) -> None:
        self.card = card
        self.out = Compiled()

    def fact(self, section: str, name: str, used_for: str, *, required: bool = True) -> Any:
        """A card value, recording its citations under ``used_for``. None if unknown."""
        sec = getattr(self.card, section)
        cited: Cited[Any] | None = getattr(sec, name) if sec is not None else None
        if cited is None or cited.value is None:
            if required:
                where = f"{section}.{name}" + ("" if sec is not None else f" ({section} missing)")
                self.out.gaps.append(f"{used_for}: needs {where}")
            return None
        self.out.provenance.setdefault(used_for, []).extend(c.model_dump() for c in cited.citations)
        return cited.value


def _id_pattern(example: str) -> str:
    prefix = example[: example.rfind("_") + 1]
    return f"^{re.escape(prefix)}[A-Za-z0-9]+$"


def _role(slot: str, default: str, intent_object: str | None) -> str:
    """Which id a path placeholder holds: by its name (`{refund_id}`, `{order_id}` when
    the intent is an order, `{payment_id}`), or the operation's default for a bare `:id`."""
    name = slot.strip(":{}").lower()
    if "refund" in name:
        return "refund_id"
    if (intent_object and intent_object.lower() in name) or "intent" in name:
        return "intent_id"
    if "payment" in name or "attempt" in name:
        return "attempt_id"
    return default


def _relative_path(
    endpoint: str,
    base_path: str,
    placeholder: str | None,
    *,
    relative: bool = False,
    intent_object: str | None = None,
) -> str | None:
    """'POST /v1/orders/:id/payments' -> 'orders/{intent_id}/payments' (None if it can't).

    ``relative``: the docs write paths relative to the base URL (Cashfree: base .../pg,
    path /orders). ``placeholder`` is the id a bare ``:id`` means; named placeholders
    keep their own role, so ``/orders/{order_id}/refunds`` is ``{intent_id}``.
    """
    path = endpoint.split(" ", 1)[1]
    if relative:
        rel = path.lstrip("/").rstrip("/")
    elif path.startswith(base_path + "/"):
        rel = path[len(base_path) + 1 :].rstrip("/")
    else:
        return None
    slots = _PLACEHOLDER.findall(rel)
    if placeholder is None:
        return None if slots else rel
    roles = [_role(s, placeholder, intent_object) for s in slots]
    if not slots or len(set(roles)) != len(roles):
        return None
    for slot, role in zip(slots, roles, strict=True):
        rel = rel.replace(slot, "{" + role + "}", 1)
    return rel


def compile_manifest(  # noqa: PLR0912, PLR0915
    card: SpecCard,
    *,
    version: str = "1",
    observed_test_key_prefix: str | None = None,
    observed_on: str = "key_id",  # which credential the observed prefix came from
    observed_errors: ErrorShapeObserved | None = None,  # from the sandbox; wins over the card
) -> Compiled:
    b = _Builder(card)
    out = b.out
    m: dict[str, Any] = {"schema_version": 1, "gateway": card.gateway, "version": version}

    # --- api + auth + credentials ---
    base = b.fact("basics", "api_base_url", "api.base_url")
    base_path = ""
    if base:
        url = urlsplit(base)
        base_path = url.path.rstrip("/")
        m["api"] = {"base_url": base.rstrip("/") + "/", "allowed_hosts": [url.hostname or ""]}
        out.notes.append(f"allowed_hosts = [{url.hostname}] from the cited base URL: confirm it")
    scheme = b.fact("basics", "auth_scheme", "auth.scheme")
    credentials: dict[str, dict[str, Any]] = {}
    if scheme == "basic":
        m["auth"] = {"scheme": "basic", "username": BASIC_USER, "password": BASIC_SECRET}
        credentials[BASIC_USER] = {"public": True}
        credentials[BASIC_SECRET] = {"secret": True}
    elif scheme == "bearer":
        m["auth"] = {"scheme": "bearer", "password": BEARER_SECRET}
        credentials[BEARER_SECRET] = {"secret": True}
    elif scheme == "api_key_headers":
        id_header = b.fact("auth_headers", "key_id_header", "auth.username_header")
        secret_header = b.fact("auth_headers", "secret_header", "auth.password_header")
        if id_header and secret_header:
            m["auth"] = {
                "scheme": "api_key_headers",
                "username": BASIC_USER,
                "password": BASIC_SECRET,
                "username_header": id_header,
                "password_header": secret_header,
            }
            credentials[BASIC_USER] = {"public": True}
            credentials[BASIC_SECRET] = {"secret": True}
        version_header = b.fact("auth_headers", "version_header", "api.headers", required=False)
        version = b.fact("auth_headers", "api_version", "api.headers", required=False)
        if version_header and version and "api" in m:
            m["api"]["headers"] = {version_header: version}
        elif version_header:
            out.gaps.append(f"api.headers: {version_header} needs auth_headers.api_version")
    elif scheme is not None:
        out.gaps.append(f"auth.scheme: {scheme!r} needs a hand-written auth scheme first")

    test_prefix = b.fact("api_keys", "test_key_prefix", "live_guard", required=False)
    live_prefix = b.fact("api_keys", "live_key_prefix", "live_guard", required=False)
    prefixed = b.fact("api_keys", "prefixed_key", "live_guard", required=False)
    if test_prefix is None and observed_test_key_prefix:
        # Not in the docs: taken from the test key a person configured for the sandbox.
        test_prefix, prefixed = observed_test_key_prefix, prefixed or observed_on
        out.provenance["live_guard"] = [
            {"ref": "observed:configured-test-key", "quote": f"test key starts {test_prefix}"}
        ]
    holder = {
        "key_id": BASIC_USER,
        "secret_key": BEARER_SECRET if scheme == "bearer" else BASIC_SECRET,
    }
    holder_name = holder.get(prefixed or "")
    if holder_name in credentials and (test_prefix or live_prefix):
        # Prefer the allowlist: a key must look like a test key (fails closed).
        m["live_guard"] = {"credential": holder_name} | (
            {"test_pattern": f"^{re.escape(test_prefix)}"}
            if test_prefix
            else {"live_pattern": f"^{re.escape(live_prefix)}"}
        )
        cred = credentials[holder_name]
        if test_prefix and live_prefix and not cred.get("secret"):
            either = "|".join(re.escape(p) for p in (test_prefix, live_prefix))
            cred["pattern"] = f"^(?:{either})[A-Za-z0-9]+$"
    else:
        out.gaps.append(
            "live_guard: no way to tell test keys from live keys (docs don't say and no "
            "test key prefix was observed); a person must add one or confirm none is possible"
        )

    # --- ids ---
    ids = {}
    for role, name in (
        ("intent", "intent_id_example"),
        ("attempt", "attempt_id_example"),
        ("refund", "refund_id_example"),
    ):
        example = b.fact("ids", name, f"ids.{role}")
        if example:
            ids[role] = _id_pattern(example)
    if len(ids) == 3:
        m["ids"] = ids

    # --- operations ---
    ops: dict[str, Any] = {}
    # Docs either repeat the base path in every endpoint (Razorpay: /v1/orders) or write
    # paths relative to the base URL (Cashfree: base .../pg, path /orders). Decide from
    # all of them together: a mix means something is off and stays a gap.
    cited = card.endpoints.cited_fields().values() if card.endpoints is not None else ()
    documented = [f.value.split(" ", 1)[1] for f in cited if f.value]
    relative = bool(base_path and documented) and not any(
        p.startswith(base_path + "/") for p in documented
    )
    if relative:
        out.notes.append(f"endpoint paths are documented relative to the base URL ({base_path})")
    intent_object = card.payment_flow.intent_object.value if card.payment_flow is not None else None

    def endpoint(
        name: str, used_for: str, placeholder: str | None, *, required: bool = True
    ) -> dict[str, str] | None:
        value = b.fact("endpoints", name, used_for, required=required)
        if value is None:
            return None
        rel = _relative_path(
            value, base_path, placeholder, relative=relative, intent_object=intent_object
        )
        if rel is None:
            out.gaps.append(
                f"{used_for}: can't place {value!r} under base path {base_path!r} "
                f"with {'its ids' if placeholder else 'no ids'}"
            )
            return None
        return {"method": value.split(" ", 1)[0], "path": rel}

    if req := endpoint("create_intent", "operations.create_payment", None):
        ops["create_payment"] = req
    if req := endpoint("get_intent", "operations.get_payment.primary", "intent_id"):
        ops["get_payment"] = {"primary": req}
        listed = endpoint(
            "list_attempts", "operations.get_payment.related.payments", "intent_id", required=False
        )
        if listed:
            ops["get_payment"]["related"] = {"payments": listed}
    manual = b.fact("capture", "manual_capture", "flags.manual_capture", required=False)
    if manual and (req := endpoint("capture", "operations.capture", "attempt_id")):
        ops["capture"] = req
    if req := endpoint("refund", "operations.refund", "attempt_id"):
        header = b.fact(
            "refunds", "idempotency_header", "operations.refund.idempotency_header", required=False
        )
        ops["refund"] = req | ({"idempotency_header": header} if header else {})
    if req := endpoint("get_refund", "operations.get_refund", "refund_id"):
        ops["get_refund"] = req
    m["operations"] = ops

    # --- flags ---
    flags = []
    if "capture" in ops:
        flags.append("manual_capture")
        if b.fact("capture", "partial_capture", "flags.partial_capture", required=False):
            flags.append("partial_capture")
    if "refund" in ops:
        if b.fact("refunds", "partial_refunds", "flags.partial_refund", required=False):
            flags.append("partial_refund")
        if b.fact("refunds", "multiple_refunds", "flags.multiple_refunds", required=False):
            flags.append("multiple_refunds")
    m["flags"] = flags

    # --- errors ---
    errors: dict[str, Any] = {}
    if msg := b.fact("error_shape", "message_path", "errors.message_paths"):
        errors["message_paths"] = [msg]
    codes = [
        b.fact("error_shape", "reason_path", "errors.code_paths", required=False),
        b.fact("error_shape", "code_path", "errors.code_paths"),
    ]
    errors["code_paths"] = [c for c in codes if c]
    if observed_errors is not None:
        _apply_observed_errors(out, errors, codes, observed_errors)
    conflict = b.fact("error_shape", "idempotency_conflict_message", "errors.rules", required=False)
    if conflict:
        errors["rules"] = [{"category": "duplicate_request", "message_contains": conflict}]
    m["errors"] = errors

    # --- webhooks ---
    if card.webhooks is not None:
        algorithm = b.fact("webhooks", "signature_algorithm", "webhooks.scheme")
        signer = b.fact("webhooks", "signing_secret", "webhooks.secrets_credential")
        header = b.fact("webhooks", "signature_header", "webhooks.signature_header")
        event_id = b.fact("webhooks", "event_id_header", "webhooks.event_id_header")
        hooks: dict[str, Any] = {}
        if algorithm == "hmac_sha256_hex":
            hooks["scheme"] = algorithm
        elif algorithm == "timestamped_hmac_sha256":
            encoding = b.fact("webhooks", "signature_encoding", "webhooks.scheme")
            stamp = b.fact("webhooks", "timestamp_header", "webhooks.timestamp_header")
            if encoding == "base64" and stamp:
                hooks |= {"scheme": "timestamped_hmac_sha256_base64", "timestamp_header": stamp}
            elif encoding == "hex":
                out.gaps.append("webhooks.scheme: timestamped hex needs a hand-written scheme")
        elif algorithm is not None:
            out.gaps.append(f"webhooks.scheme: {algorithm!r} needs a hand-written scheme first")
        if signer == "webhook_secret":
            credentials[WEBHOOK_SECRETS] = {"secret": True, "multiple": True, "required": False}
            hooks["secrets_credential"] = WEBHOOK_SECRETS
        elif signer == "api_key_secret" and m.get("auth", {}).get("password"):
            hooks["secrets_credential"] = m["auth"]["password"]  # signed with the API secret
            out.notes.append("webhooks are signed with the API secret: rotating it rotates both")
        elif signer is not None:
            out.gaps.append(f"webhooks: signed with {signer!r} but the auth has no such secret")
        if "scheme" in hooks and "secrets_credential" in hooks and header and event_id:
            m["webhooks"] = {
                "scheme": hooks["scheme"],
                "signature_header": header,
                "event_id_header": event_id,
                "secrets_credential": hooks["secrets_credential"],
            } | (
                {"timestamp_header": hooks["timestamp_header"]}
                if "timestamp_header" in hooks
                else {}
            )
    else:
        out.notes.append("no webhooks: the card has no webhooks section")

    # --- checkout return (hosted checkout only) ---
    completion = b.fact("payment_flow", "customer_completion", "checkout_return", required=False)
    if completion == "hosted_checkout" and card.checkout_return is not None and scheme == "basic":
        cr = {
            n: b.fact("checkout_return", n, "checkout_return")
            for n in (
                "intent_param",
                "attempt_param",
                "signature_param",
                "signed_message",
                "signature_scheme",
                "signing_key",
            )
        }
        if None in cr.values():
            pass  # gaps already recorded
        elif "other" in (cr["signed_message"], cr["signature_scheme"], cr["signing_key"]):
            out.gaps.append("checkout_return: unsupported signing; needs a hand-written scheme")
        else:
            parts = [cr["intent_param"], cr["attempt_param"]]
            if cr["signed_message"] == "payment_id|intent_id":
                parts.reverse()
            m["checkout_return"] = {
                "scheme": cr["signature_scheme"],
                "message_template": "|".join("{" + p + "}" for p in parts),
                "key_credential": BASIC_SECRET,
                "signature_param": cr["signature_param"],
                "intent_param": cr["intent_param"],
            }

    m["credentials"] = credentials
    out.notes.append("no [settings]: gateway-specific settings are added by a person if needed")
    out.manifest = {k: m[k] for k in _ORDER if k in m}
    if not out.gaps:
        try:
            Manifest.model_validate(out.manifest)
        except ValidationError as exc:
            out.problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return out


def _apply_observed_errors(
    out: Compiled, errors: dict[str, Any], codes: list[Any], seen: ErrorShapeObserved
) -> None:
    """What the sandbox showed beats a reading of the docs; each change is noted."""
    reason, code = codes
    observed = {
        "message_paths": [seen.message_path],
        "code_paths": [seen.reason_path, seen.code_path],
    }
    card = {"message_paths": errors.get("message_paths", []), "code_paths": [reason, code]}
    for key, values in observed.items():
        if None in values:
            out.notes.append(f"errors.{key}: the sandbox didn't settle it; kept the card's")
            continue
        if values != card[key]:
            out.notes.append(f"errors.{key}: card said {card[key]}, sandbox shows {values}")
            out.gaps[:] = [g for g in out.gaps if not g.startswith(f"errors.{key}")]
        errors[key] = values
        out.provenance[f"errors.{key}"] = [
            {"ref": "observed:sandbox", "quote": f"{p} = {seen.evidence.get(p or '')}"}
            for p in values
        ]


_ORDER = (
    "schema_version",
    "gateway",
    "version",
    "flags",
    "api",
    "auth",
    "credentials",
    "live_guard",
    "ids",
    "operations",
    "errors",
    "webhooks",
    "checkout_return",
)


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            out |= flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(node, list) and node and isinstance(node[0], dict):
        out = {}
        for i, v in enumerate(node):
            out |= flatten(v, f"{prefix}[{i}]")
        return out
    if isinstance(node, list):
        return {prefix: sorted(map(str, node))}
    return {prefix: node}


def compare(
    generated: dict[str, Any],
    reference: dict[str, Any],
    *,
    ignore: tuple[str, ...] = ("settings.",),
) -> dict[str, list[str]]:
    """Field-level comparison with a hand-written manifest (field accuracy, PLAN.md §10)."""
    gen, ref = flatten(generated), flatten(reference)
    keys = sorted(k for k in set(gen) | set(ref) if not k.startswith(ignore))
    result: dict[str, list[str]] = {"same": [], "different": [], "missing": [], "extra": []}
    for k in keys:
        if k not in gen:
            result["missing"].append(f"{k} = {ref[k]!r}")
        elif k not in ref:
            result["extra"].append(f"{k} = {gen[k]!r}")
        elif gen[k] == ref[k]:
            result["same"].append(k)
        else:
            result["different"].append(f"{k}: generated {gen[k]!r}, reference {ref[k]!r}")
    return result


def test_key_prefix(key: str) -> str | None:
    """'rzp_test_AbC123' -> 'rzp_test_'. None unless the prefix says it's a test key.

    Only the prefix is kept; the key itself is never stored or shown.
    """
    # Cut right after the "test" segment: what follows can be secret. Cashfree's secret
    # (cfsk_ma_test_<secret>_<more>) has underscores after it; cutting at the last one
    # would copy part of the secret into the manifest.
    parts = key.split("_")
    marker = next((i for i, part in enumerate(parts[:-1]) if part.lower() == "test"), None)
    return "_".join(parts[: marker + 1]) + "_" if marker is not None else None
