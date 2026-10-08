"""Gate G8: a person approves the exact package bytes, and only approved packages run.

A package directory holds ``manifest.toml`` + ``mappers.py`` (whose sha256 is
``GatewayPackage.sha256``) and, once approved, ``approval.toml``: an Ed25519
signature over that hash plus who signed and when. ``packages/trusted_signers.toml``
lists the public keys allowed to approve; removing an entry revokes it.

Private keys live outside the repo (``~/.config/payweave/keys`` or
``$PAYWEAVE_KEY_DIR``), encrypted with a passphrase. That is the point: a tool
with shell access (including the agent that wrote the package) can read the
key file but cannot sign without the person who knows the passphrase.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from payweave.packages.runtime import (
    GatewayPackage,
    PackageLoadError,
    load_package_from_sources,
    package_hash,
)

APPROVAL_FILE = "approval.toml"
DEFAULT_TRUST_FILE = Path("packages/trusted_signers.toml")
_DOMAIN = b"payweave-package-approval-v1\n"  # signatures can't be replayed as anything else
MIN_PASSPHRASE = 12

SignerName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ApprovalError(PackageLoadError):
    """The package is unsigned, changed since it was signed, or signed by an untrusted key."""


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    gateway: str
    version: str
    package_sha256: Sha256Hex
    signer: SignerName
    signed_at: datetime
    signature: str  # base64 Ed25519 over _DOMAIN + the other fields

    def message(self) -> bytes:
        fields = self.model_dump(mode="json", exclude={"signature"})
        return _DOMAIN + json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()

    def to_toml(self) -> str:
        # Every value is pattern-checked or ISO/base64, so JSON string quoting is valid TOML.
        lines = [f"{k} = {json.dumps(v)}" for k, v in self.model_dump(mode="json").items()]
        return (
            "# Written by `python -m payweave.packages sign`. Do not edit.\n"
            + "\n".join(lines)
            + "\n"
        )


# --- keys ------------------------------------------------------------------------------------


def key_dir() -> Path:
    return Path(os.environ.get("PAYWEAVE_KEY_DIR", "~/.config/payweave/keys")).expanduser()


def public_key_b64(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:  # gitleaks:allow (a type)
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def write_private_key(name: str, passphrase: str) -> tuple[Path, str]:
    """Create ``<key_dir>/<name>.pem`` (0600, passphrase-encrypted). Returns path, public key."""
    if len(passphrase) < MIN_PASSPHRASE:
        raise ValueError(f"passphrase must be at least {MIN_PASSPHRASE} characters")
    directory = key_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{name}.pem"
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(passphrase.encode()),
    )
    # O_EXCL: never overwrite an existing key.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return path, public_key_b64(key)


def read_private_key(name: str, passphrase: str) -> Ed25519PrivateKey:
    path = key_dir() / f"{name}.pem"
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), passphrase.encode())
    except FileNotFoundError as exc:
        raise ApprovalError(f"no signing key at {path}; run keygen first") from exc
    except ValueError as exc:  # wrong passphrase or not a key
        raise ApprovalError("could not unlock the signing key (wrong passphrase?)") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ApprovalError(f"{path} is not an Ed25519 key")
    return key


# --- trust -----------------------------------------------------------------------------------


def load_trust(path: Path = DEFAULT_TRUST_FILE) -> dict[str, Ed25519PublicKey]:
    """Signer name -> public key. A missing file trusts nobody."""
    if not path.exists():
        return {}
    signers = tomllib.loads(path.read_text()).get("signers", {})
    trusted: dict[str, Ed25519PublicKey] = {}
    for name, entry in signers.items():
        try:
            trusted[name] = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(entry["public_key"], validate=True)
            )
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise ApprovalError(f"{path}: bad public key for signer {name!r}") from exc
    return trusted


def add_trusted_signer(name: str, public_key: str, path: Path = DEFAULT_TRUST_FILE) -> None:
    current = tomllib.loads(path.read_text()).get("signers", {}) if path.exists() else {}
    if name in current:
        raise ValueError(f"signer {name!r} is already in {path}")
    header = (
        ""
        if path.exists()
        else "# Public keys allowed to approve Gateway Packages (gate G8).\n"
        "# Remove an entry to revoke it; its approvals stop loading.\n"
    )
    added = datetime.now(UTC).date().isoformat()
    with path.open("a") as f:
        f.write(f'{header}\n[signers.{name}]\npublic_key = "{public_key}"\nadded = "{added}"\n')


# --- sign / verify ---------------------------------------------------------------------------


def sign_package(
    package: GatewayPackage, key: Ed25519PrivateKey, signer: str, *, now: datetime | None = None
) -> Approval:
    unsigned = Approval(
        gateway=package.manifest.gateway,
        version=package.manifest.version,
        package_sha256=package.sha256,
        signer=signer,
        signed_at=(now or datetime.now(UTC)).replace(microsecond=0),
        signature="",
    )
    signature = base64.b64encode(key.sign(unsigned.message())).decode()
    return unsigned.model_copy(update={"signature": signature})


def verify_approval(
    package: GatewayPackage, approval: Approval, trust: dict[str, Ed25519PublicKey]
) -> None:
    m = package.manifest
    if (approval.gateway, approval.version) != (m.gateway, m.version):
        raise ApprovalError(
            f"approval is for {approval.gateway} v{approval.version}, "
            f"package is {m.gateway} v{m.version}"
        )
    if approval.package_sha256 != package.sha256:
        raise ApprovalError(
            f"{m.gateway} v{m.version} changed since it was approved "
            f"(approved {approval.package_sha256[:12]}, now {package.sha256[:12]})"
        )
    _check_signature(approval, trust)


def _check_signature(approval: Approval, trust: dict[str, Ed25519PublicKey]) -> None:
    public = trust.get(approval.signer)
    if public is None:
        raise ApprovalError(f"signer {approval.signer!r} is not trusted")
    try:
        public.verify(base64.b64decode(approval.signature, validate=True), approval.message())
    except (InvalidSignature, ValueError, binascii.Error) as exc:
        raise ApprovalError(
            f"bad signature on the {approval.gateway} v{approval.version} approval"
        ) from exc


def read_approval(directory: Path) -> Approval:
    path = directory / APPROVAL_FILE
    try:
        return Approval.model_validate(tomllib.loads(path.read_text()))
    except FileNotFoundError as exc:
        raise ApprovalError(f"{directory} has no {APPROVAL_FILE}: it was never approved") from exc
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ApprovalError(f"{path} is malformed: {exc}") from exc


def load_signed_package(
    directory: Path, trust: dict[str, Ed25519PublicKey] | None = None
) -> GatewayPackage:
    """Load a package only if a trusted signer approved these exact bytes.

    The hash and signature are checked *before* the mappers are executed, so
    unapproved code never runs, not even in the sandbox.
    """
    trust = load_trust() if trust is None else trust
    approval = read_approval(directory)
    manifest_bytes = (directory / "manifest.toml").read_bytes()
    mappers_bytes = (directory / "mappers.py").read_bytes()
    sha256 = package_hash(manifest_bytes, mappers_bytes)
    if sha256 != approval.package_sha256:
        raise ApprovalError(
            f"{directory} changed since it was approved "
            f"(approved {approval.package_sha256[:12]}, now {sha256[:12]})"
        )
    _check_signature(approval, trust)
    package = load_package_from_sources(
        manifest_bytes.decode(), mappers_bytes.decode(), path=directory
    )
    verify_approval(package, approval, trust)  # also binds gateway + version
    return replace(package, approval=approval)
